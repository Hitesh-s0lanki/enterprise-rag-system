# Repository instructions

These instructions apply to the entire repository. Read any more specific `AGENTS.md` before changing files in its directory.

## Python environment and dependencies

- Use `uv` for Python environment and package management.
- Use this repository's `.venv` for all Python scripts, application commands, tests, and Python-based tools. Do not install or run project dependencies through system Python, global pip, or another project's environment.
- Run commands from the repository root. Respect `.python-version` and the Python requirement in `pyproject.toml`.
- Initialize or synchronize the local environment with `uv sync --locked`. If `.venv` must be created explicitly, use `uv venv .venv --python 3.13`, then synchronize it.
- Run Python with `uv run --locked python ...` or `.venv/bin/python ...`. Run installed tools through `uv run --locked <tool> ...`.
- Add runtime packages with `uv add <package>` and development packages with `uv add --dev <package>`. Remove packages with `uv remove <package>` (or `uv remove --dev <package>` for development dependencies).
- Declare every directly used third-party package in `pyproject.toml`; do not rely on a package being available only as a transitive dependency or installed manually in `.venv`.
- Keep `pyproject.toml`, `uv.lock`, and `requirements.txt` synchronized whenever dependencies change. `pyproject.toml` declares dependencies, `uv.lock` records resolved versions, and `requirements.txt` is the readable, pinned export for reference and interoperability.
- After adding, removing, or updating a package, regenerate `requirements.txt` with:

  ```sh
  uv export --locked --all-groups --no-hashes --no-emit-project --output-file requirements.txt
  ```

- Do not manually edit the generated requirements list. Preserve its dependency annotations so the reason each package is present can be inspected.
- Standard-library modules, including `sqlite3`, do not need a third-party dependency entry.
- Keep `.venv` out of version control. Never add real credentials to dependency files, commands, or logs.

## Project structure and constraints

- Keep the problem statement in `docs/problem-statement.md` focused on business needs and requirements.
- Put explanatory documentation in `/docs`; do not add README files inside source packages.
- Put shared external-service clients in `src/clients`.
- Keep text loading, recursive character splitting, embedding preparation, and ingestion orchestration in `src/data_ingestion`.
- Keep auth models, SQLite storage, repository adapters, and authorization under `src/db/auth`.
- New agents belong in `src/agents` and inherit `BaseAgent` from `src/agents/base_agent.py`. Keep shared instructions in `src/agents/prompts/base_agent.md` and each agent's role instructions in `src/agents/prompts/<prompt_name>.md`; do not embed role prompts in Python code.
- Put agent tools in `src/tools` and register root tools explicitly in `src/tools/registry.py`. Keep trusted user identity and approved revisions outside model-controlled tool arguments. Preserve the `AgentEvent` contract when changing root streaming, and verify tool failures, cancellation and terminal-event behavior.
- Keep shared runtime/stream orchestration in `BaseAgent`; child agents should supply tools and authorization hooks. Put event definitions, serialization and callback-normalization helpers in `src/utils`, not inside individual agent modules.
- Use LangChain for agent work and Streamlit for the showcase frontend. Follow the agreed architecture before introducing other frameworks.
- Use OpenAI for embeddings and answer generation, Pinecone for hybrid-ready document storage, Redis for semantic caching, and TypeSafe AI Jev for document evidence grading when those components are implemented.
- Preserve filename, source update time, ingestion time, last editor, uploader identity, role metadata, and document/chunk revision identity through ingestion.

## Database and access control

- Use standard-library `sqlite3` for current local auth storage. Read users, role permission masks, and namespace scopes from the database rather than hardcoding lookup results in repositories or services.
- Initial seed data belongs in `src/db/auth/seed.json`. Initialization must be repeatable without resetting existing edits, deleted/deactivated users, scopes, or permission changes.
- Keep storage behind the repository interface so a PostgreSQL adapter can replace SQLite later. Use parameterized SQL, database constraints, and transactions for related writes.
- Enforce bitwise action permissions and namespace/document access in services before side effects. Role metadata and caller-supplied usernames alone do not establish authorization.
- Keep source editor identity separate from the acting uploader. The demo CLI user selector is not production authentication.
- Existing-document updates require authoritative ACL checks and revision publication; do not imply that raw Pinecone upserts implement that workflow.

## Configuration and verification

- Keep `.env` private. Document supported environment variables with safe placeholders and defaults in `.env.example` whenever configuration changes.
- Keep local SQLite databases and journal files out of version control.
- Prefer offline tests and dry runs for routine verification. Do not use real provider calls just to test imports, metadata handling, or permission checks.
- For auth or ingestion behavior changes, run:

  ```sh
  uv run --locked python -m unittest discover -s tests -v
  ```

- Check allowed and denied operations when changing permissions. Verify that denied ingestion does not call embedding or storage providers.
- Before reporting completion, state what changed, what was verified, and any remaining functional limitation. Never claim an external upload or database change happened without verifying it.
