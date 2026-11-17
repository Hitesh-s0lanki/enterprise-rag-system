# Agent Foundation

The main tool-calling implementation is now [RootAgent](root-agent.md), built on the base class described below. Its tools live under `src/tools`, and its role prompt is `src/agents/prompts/root_agent.md`.

New agents inherit `BaseAgent` from `src/agents/base_agent.py`. The base class owns model/run configuration, prompt loading, runtime construction, call limits, sync/async execution, streaming orchestration, cancellation, cleanup and SSE wrappers. OpenAI chat-client construction lives in `src/clients/openai_client.py` and uses LangChain's `ChatOpenAI`.

Event definitions and JSON/SSE serialization live in `src/utils/events.py`. Per-run event IDs, usage aggregation, callback normalization, safe error conversion and cancellable event waiting live in `src/utils/streaming.py`. `RootAgent` only binds its service/tools and supplies its authorization hook and root-specific settings.

## Prompt files

- `src/agents/prompts/base_agent.md`: shared instructions loaded for every agent.
- `src/agents/prompts/<prompt_name>.md`: each child agent's role-specific instructions.

Set `prompt_name` on the child class without the `.md` extension. Names allow lowercase letters, digits and underscores, starting with a letter. Missing or empty files fail during agent construction. Files resolving outside the prompt directory are rejected. Markdown is loaded literally, so JSON braces are not interpreted as template variables. Keep role instructions in these files rather than Python strings.

## Inheritance example

When adding a future answer agent, first create its instructions in `src/agents/prompts/answer_agent.md`, then inherit the common execution methods:

```python
from src.agents import BaseAgent


class AnswerAgent(BaseAgent):
    prompt_name = "answer_agent"
```

This example does not create an answer agent in the repository yet. `BaseAgent` is abstract until a subclass defines `prompt_name`. The child then inherits `run`, `arun`, `stream`, `astream`, `stream_sse` and `astream_sse` without implementing them. `run` and `arun` return the final answer/run metrics dictionary. Subclasses can override `get_tools()` to supply registered tools, `_authorize()` to enforce service access, or `create_runtime()` for a specialized runtime. Protected-data agents must supply an authorization hook; the generic default is a no-op. `RootAgent` supplies the existing knowledge authorization checks.

The model helpers return the full LangChain message, preserving `usage_metadata` and provider response metadata. They accept `runnable_config` for LangChain callbacks, tags and tracing configuration. Reference context is passed as human-message data rather than promoted to system instructions.

## Configuration

Entry points should load `.env` before constructing agents; the base class and client factory do not load it implicitly. Pass `AgentConfig(...)` to override environment-derived configuration, or inject a `BaseChatModel` through `llm=` for testing or application-managed model reuse. Client creation is lazy: constructing an agent and inspecting its prompts does not need credentials or make provider requests.

| Setting | Default | Purpose |
| --- | --- | --- |
| `AGENT_MODEL` | `gpt-4.1-mini` | Configurable OpenAI chat model. |
| `AGENT_TEMPERATURE` | `none` | Omit a temperature value; set a number only for models that support it. |
| `AGENT_MAX_OUTPUT_TOKENS` | `1024` | Output limit per model invocation. |
| `AGENT_TIMEOUT_SECONDS` | `30` | Provider request timeout. |
| `AGENT_MAX_RETRIES` | `2` | Provider retry limit. |
| `AGENT_MAX_INPUT_CHARACTERS` | `16000` | Combined query/context character limit. |
| `AGENT_RUN_MAX_MODEL_CALLS` | `4` | Shared model-call limit for future agents. |
| `AGENT_RUN_MAX_TOOL_CALLS` | `6` | Shared tool-call limit for future agents. |
| `AGENT_RUN_TIMEOUT_SECONDS` | `120` | Shared overall execution timeout. |

`AgentRunSettings` validates common execution limits. Pass it as `settings=` to override environment defaults. `RootAgentSettings` reuses that validation while reading the existing `ROOT_AGENT_*` variables independently. `AGENT_TIMEOUT_SECONDS` remains the timeout for an individual provider request.

Live OpenAI calls require `OPENAI_API_KEY`. Provider-specific model limits and supported parameters still apply. Character/output limits are basic request controls, not a complete workflow token or monetary budget. Input validation occurs before client creation.

The common runtime executes the model/tool loop; subclasses supply the tools and access policy. Retrieval, Jev grading and caching remain in their existing services, called through registered tools. Shared prompts do not replace service-level access checks. `AgentRunError` is the common sanitized run error; `RootAgentError` remains an alias for existing callers.

Verification uses fake LangChain models and mocked client construction; no live model calls are needed:

```sh
uv run --locked python -m unittest discover -s tests -v
```

[LangChain ChatOpenAI documentation](https://docs.langchain.com/oss/python/integrations/chat/openai) describes the underlying model interface and response metadata.
