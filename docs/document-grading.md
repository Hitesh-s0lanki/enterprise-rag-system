# Query-based document grading

`src.grading.DocumentGradingService` evaluates the chunks returned by hybrid
retrieval against the user's query using the shared Jev client. Each chunk gets
one request containing three questions: relevance, coverage, and sufficient
evidence. It does not generate an answer or grade a generated answer.

Run retrieval and grading together with the maintainer CLI:

```sh
uv run --locked python -m src.retrieval "How do I connect to the VPN?" \
  --user alice --revision DOCUMENT_SHA256:REVISION_SHA256 --grade
```

Replace the revision placeholders with approved IDs from your ingestion output.
The existing provider keys and `TYPESAFE_API_KEY` must be configured. This command
makes live embedding, search, and Jev calls. The demo user selector is not
production authentication.

The output retains the retrieval result and adds a `grading` object with:

- `query`, `rubric_version`, `rubric`, and decision thresholds.
- `documents` in grade order, each containing original and revised ranks,
  source text, complete source/revision metadata, normalized relevance and
  coverage scores, sufficient-evidence probability, and a decision.
- Each document's complete typed Jev evaluation, including model, raw scores,
  confidence, probability distributions, rubric legends, and token usage.
- `summary` with evaluated, relevant, and accepted chunk counts, a best chunk ID,
  and `no_documents`, `sufficient_evidence`, or `insufficient_evidence` status.
- `metrics` with total latency, logical Jev call count, and actual token usage.
  SDK retries may create additional HTTP attempts. Cost is unknown (`null`).

Add `--memory-cache` to reuse a valid previous report through Mem0 before running
retrieval and grading. See [query result reuse](query-memory-cache.md) for the
output envelope, access checks, expiry, and paraphrase verification.

The service uses two five-level Score questions. Jev scores range from 0 to 4;
the service divides by 4 for normalized 0–1 values. These scores are not
probabilities. The Noul sufficient-evidence value is a probability. See the
[Score documentation](https://docs.typesafe.ai/primitives/score).
The report includes raw answers so uncertainty remains visible; it does not
invent natural-language model explanations.

Ranking sorts by relevance, then coverage, then sufficient-evidence probability,
all descending. Ties preserve retrieval order. A chunk is accepted when relevance
is at least `GRADING_RELEVANCE_THRESHOLD` and sufficient-evidence probability is
at least `GRADING_EVIDENCE_THRESHOLD`. All chunks remain in the report, including
rejected ones. The summary indicates whether at least one individual chunk passes;
it does not assess whether several partial chunks jointly answer the query or
whether chunks contradict one another. Defaults are initial heuristics requiring
calibration on your own labeled queries.

Connection options stay in `JevSettings`; grading limits and thresholds live in
`GradingSettings` and are documented in `.env.example`. Explicit settings may be
injected independently for each use case:

```python
from src.clients.jev_client import create_jev_client
from src.grading import DocumentGradingService, GradingSettings

with create_jev_client() as jev:
    grader = DocumentGradingService(
        client=jev, authorization=authorization, namespace="it-support",
        settings=GradingSettings(relevance_threshold=0.75),
    )
    report = grader.grade(
        query, retrieval_result.documents,
        actor_id=user.id, approved_revisions=approved_revisions,
    ).to_dict()
```

The service preflights permissions, roles, approved revisions, IDs, counts, and
text sizes for the entire input before Jev calls. It rechecks access before each
request and before returning the report. Invalid input, denied access, provider
errors, and invalid grading values propagate; no silently partial success report
is returned. An empty input produces a report without calling Jev. Text is never
silently truncated. Earlier calls may already have happened if a later provider
request fails or access changes during grading.

Only trusted retrieval code should supply document metadata, namespace, actor ID,
and approved revision maps. This follows the current retrieval trust boundary;
caller-written role metadata is not authoritative authorization. A production
document ACL and revision registry remain necessary. The report contains document
text and must retain the same access restrictions as the source material.
