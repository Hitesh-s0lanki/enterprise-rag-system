"""Reuse complete, authorized reports from Mem0 before running a lazy pipeline."""

import hashlib
import json
import math
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from time import perf_counter, time
from uuid import UUID

from src.clients.mem0_client import Mem0Client, Mem0Error
from src.db.auth import AuthorizationService, Permission
from src.retrieval.service import validate_revisions
from src.utils.logger import current_run_id, get_logger

from .config import QueryCacheSettings
from .equivalence import query_equivalence

CACHE_SCHEMA = "rag-report-cache-v2"
logger = get_logger(__name__)


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class MemoryFirstQueryService:
    def __init__(self, *, memory: Mem0Client, authorization: AuthorizationService,
                 namespace: str, pipeline_config: dict, settings: QueryCacheSettings | None = None,
                 equivalence: Callable[[str, str], float] = query_equivalence):
        self.memory = memory
        self.authorization = authorization
        self.namespace = namespace
        self.pipeline_config = deepcopy(pipeline_config)
        self.settings = settings or QueryCacheSettings()
        self.equivalence = equivalence

    def _check_report(self, report: dict, actor_id: UUID, revisions: dict[str, str]):
        """Recheck every returned chunk, including rejected grading candidates."""
        self.authorization.require_access(actor_id, Permission.READ_KNOWLEDGE, namespace=self.namespace)
        if not isinstance(report, dict):
            raise ValueError("Report must be a JSON object")
        rows = report["documents"]
        graded = report["grading"]["documents"]
        if not isinstance(rows, list) or not rows or not isinstance(graded, list):
            raise ValueError("Only nonempty, graded reports are reusable")
        if report["grading"]["summary"]["status"] != "sufficient_evidence":
            raise ValueError("Insufficient evidence reports are not reusable")
        by_id = {row["chunk_id"]: row for row in rows}
        if (len(by_id) != len(rows) or len(graded) != len(rows)
            or {row["chunk_id"] for row in graded} != set(by_id)):
            raise ValueError("Report contains duplicate or mismatched chunks")
        for row in rows + graded:
            if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
                raise ValueError("Invalid cached document")
            metadata = row["metadata"]
            roles = metadata.get("allowed_roles")
            if not isinstance(roles, list) or not roles:
                raise PermissionError("Missing cached document access roles")
            if (revisions.get(metadata.get("document_id")) != metadata.get("revision_id")
                or metadata.get("document_id") not in revisions
                or metadata.get("chunk_id") != row["chunk_id"]):
                raise ValueError("Cached report contains an unapproved revision or invalid ID")
            original = by_id.get(row["chunk_id"])
            if original is None or original["text"] != row["text"] or original["metadata"] != metadata:
                raise ValueError("Cached grading and retrieval chunks disagree")
            self.authorization.require_access(actor_id, Permission.READ_KNOWLEDGE,
                                              namespace=self.namespace, allowed_roles=roles)

    def run(self, query: str, *, actor_id: UUID, approved_revisions: dict[str, str],
            pipeline: Callable[[], dict]) -> dict:
        """The callback must create provider clients only when called on a miss.

        Callers supply trusted current revision approvals and pipeline settings.
        Memory metadata is trusted application storage, never an end-user input.
        """
        start = perf_counter()
        timings = {}

        def measured(stage, operation, *args, **kwargs):
            began = perf_counter()
            try:
                return operation(*args, **kwargs)
            finally:
                timings[stage] = timings.get(stage, 0) + (perf_counter() - began) * 1000

        user = self.authorization.require_access(actor_id, Permission.READ_KNOWLEDGE, namespace=self.namespace)
        if not isinstance(query, str) or not query.strip() or len(query) > 4000:
            raise ValueError("Query must contain 1-4000 characters")
        query = query.strip()
        revisions = validate_revisions(approved_revisions)
        context = {
            "schema": CACHE_SCHEMA, "actor_id": str(actor_id), "role": user.role.value,
            "namespace": self.namespace, "revisions": revisions, "pipeline": self.pipeline_config,
        }
        context_id = hashlib.sha256(_json(context).encode()).hexdigest()
        # Restrict candidates before ranking. The live v3 endpoint ignores the
        # top-level metadata option and otherwise ranks report payloads above
        # query entries, filling top_k with records we cannot reuse.
        cutoff = datetime.fromtimestamp(time() - self.settings.ttl_seconds, timezone.utc).isoformat()
        filters = {"AND": [
            {"user_id": str(actor_id)}, {"run_id": context_id},
            {"metadata": {"schema": CACHE_SCHEMA, "cache_kind": "query"}},
            {"created_at": {"gte": cutoff}},
        ]}
        search_state = "miss"
        equivalence_calls = 0
        try:
            response = measured("memory_search_latency_ms", self.memory.search, query,
                                filters=filters, top_k=self.settings.top_k, rerank=False)
            candidates = response.get("results", [])
            if not isinstance(candidates, list):
                raise Mem0Error("Mem0 returned malformed search results")
        except Mem0Error:
            candidates = []
            search_state = "unavailable"
        # Prioritize exact repeats so unrelated candidates cannot trigger extra model calls.
        candidates.sort(key=lambda item: not (isinstance(item, dict) and item.get("memory") == query))
        valid = []
        for candidate in candidates:
            try:
                if not isinstance(candidate, dict) or not isinstance(candidate.get("metadata"), dict):
                    continue
                score = candidate["score"]
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                    continue
                if not 0 <= score <= 1:
                    continue
                if score < self.settings.similarity_threshold and candidate.get("memory") != query:
                    continue
                metadata = candidate["metadata"]
                if (metadata["schema"] != CACHE_SCHEMA or metadata["context_id"] != context_id
                    or metadata.get("cache_kind") != "query"):
                    continue
                if metadata["actor_id"] != str(actor_id) or metadata["namespace"] != self.namespace:
                    continue
                created = metadata["stored_at"]
                if isinstance(created, bool) or not isinstance(created, (int, float)) or not math.isfinite(created):
                    continue
                age = time() - created
                if not 0 <= age < self.settings.ttl_seconds:
                    continue
                saved_query = candidate["memory"]
                if not isinstance(saved_query, str) or not saved_query.strip() or len(saved_query) > 4000:
                    continue
                exact = query == saved_query
                probability = 1.0
                if not exact:
                    equivalence_calls += 1
                    try:
                        probability = measured("equivalence_latency_ms", self.equivalence, query, saved_query)
                    except Exception:
                        # An unavailable verifier can only cause a miss, never unsafe reuse.
                        continue
                    if not math.isfinite(probability) or not self.settings.equivalence_threshold <= probability <= 1:
                        continue
                stored = measured("memory_fetch_latency_ms", self.memory.get, metadata["report_memory_id"])
                stored_metadata = stored.get("metadata")
                if not isinstance(stored_metadata, dict):
                    continue
                if any(stored_metadata.get(key) != metadata[key]
                       for key in ("schema", "context_id", "actor_id", "namespace", "report_sha256", "stored_at")):
                    continue
                if stored_metadata.get("cache_kind") != "report":
                    continue
                payload = stored["memory"]
                if not isinstance(payload, str) or len(payload.encode()) > self.settings.max_payload_bytes:
                    continue
                if hashlib.sha256(payload.encode()).hexdigest() != metadata["report_sha256"]:
                    continue
                report = json.loads(payload)
                self._check_report(report, actor_id, revisions)
                if report["grading"]["query"] != saved_query:
                    continue
                valid.append((score, created, candidate.get("id"), saved_query, report, probability))
                if exact:
                    break
            except Mem0Error:
                search_state = "unavailable"
            except (KeyError, ValueError, TypeError, PermissionError):
                continue
        if valid:
            score, created, memory_id, matched_query, report, probability = max(valid, key=lambda x: (x[0], x[1]))
            # Recheck user and document access after the network lookup and before returning.
            self._check_report(report, actor_id, revisions)
            return self._output(query, report, start, hit=True, equivalence_calls=equivalence_calls,
                                timings=timings,
                                status="hit", memory_id=memory_id, equivalence_probability=probability,
                                similarity=score, matched_query=matched_query, stored_at=created)

        report = measured("pipeline_latency_ms", pipeline)
        # Cache failures must never discard a valid pipeline result.
        write_state = "skipped"
        reusable = False
        try:
            self._check_report(report, actor_id, revisions)
            reusable = True
            payload = _json(report)
            if len(payload.encode()) <= self.settings.max_payload_bytes:
                common = {
                    "schema": CACHE_SCHEMA, "context_id": context_id,
                    "actor_id": str(actor_id), "namespace": self.namespace, "stored_at": time(),
                    "report_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                }
                scope = {"user_id": str(actor_id), "run_id": context_id, "infer": False, "immutable": True}
                result = measured("memory_write_latency_ms", self.memory.add, payload,
                                  **scope, metadata={**common, "cache_kind": "report"})
                write_state = self._write_status(result)
                if write_state == "stored":
                    report_id = result["results"][0]["id"]
                    result = measured("memory_write_latency_ms", self.memory.add, query, **scope, metadata={
                        **common, "cache_kind": "query", "report_memory_id": report_id,
                    })
                    write_state = self._write_status(result)
        except Mem0Error:
            write_state = "failed"
        except (KeyError, TypeError, ValueError):
            write_state = "skipped"
        # Do not swallow current authorization denial on a pipeline result.
        self.authorization.require_access(actor_id, Permission.READ_KNOWLEDGE, namespace=self.namespace)
        if reusable:
            self._check_report(report, actor_id, revisions)
        return self._output(query, report, start, hit=False, equivalence_calls=equivalence_calls,
                            timings=timings,
                            status=search_state, write_status=write_state)

    @staticmethod
    def _write_status(result):
        if result.get("status") == "SUCCEEDED" and result.get("results"):
            return "stored"
        if result.get("status") in ("PENDING", "RUNNING"):
            return "pending"
        return "unconfirmed"

    @staticmethod
    def _output(query, report, start, *, hit, equivalence_calls, timings=None, **cache):
        output = {
            "query": query, "source": "mem0" if hit else "rag_pipeline", "result": report,
            "cache": {"hit": hit, **cache},
            "execution": {"pipeline_runs": 0 if hit else 1,
                          "equivalence_checks": equivalence_calls,
                          **(timings or {}),
                          "total_latency_ms": (perf_counter() - start) * 1000},
        }
        logger.info("knowledge.lookup_completed", extra={
            "run_id": current_run_id(), "cache_source": output["source"],
            "cache_status": cache.get("status"), "cache_write_status": cache.get("write_status"),
            **output["execution"],
        })
        return output
