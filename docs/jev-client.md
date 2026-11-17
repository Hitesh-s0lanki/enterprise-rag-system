# Shared Jev client

Import `create_jev_client` from `src.clients.jev_client` wherever Jev is needed.
Connection configuration lives in this module so future changes have one place
to go. The factory returns the official TypeSafe SDK client; typed responses,
question primitives, error handling, and per-request overrides remain available.
See the [TypeSafe quickstart](https://docs.typesafe.ai/introduction/quickstart)
and [SDK reference](https://docs.typesafe.ai/sdk/python/api/clients/sync).

Set `TYPESAFE_API_KEY` and optionally the settings shown in `.env.example`.
The factory reads the current environment on each construction. Application
entry points are responsible for loading `.env`, if needed; importing this module
does not load credentials or create a connection.

```python
from dotenv import load_dotenv
from typesafe_sdk import Noul

from src.clients.jev_client import create_jev_client

load_dotenv()  # At the application entry point.

with create_jev_client() as client:
    result = client.system_one(
        state={
            "question": "How do employees connect to the VPN?",
            "document": "Employees connect to the VPN using MFA.",
        },
        questions={
            "relevant": Noul(
                instructions="The document provides evidence to answer the question.",
            ),
        },
    )
    relevance_probability = result.nouls["relevant"].noul
```

For different connection requirements, pass explicit settings:

```python
from src.clients.jev_client import JevSettings, create_jev_client

settings = JevSettings(timeout_seconds=10, max_retries=0)
with create_jev_client(settings) as client:
    pass  # Reuse this client for multiple evaluations.
```

Explicit settings replace environment settings as a whole. An optional
`api_key=` overrides `TYPESAFE_API_KEY`; keys are never stored in `JevSettings`.
The default model is `jev-latest`, the API root is `https://api.typesafe.ai`,
the HTTP operation timeout is 30 seconds, and the maximum retry count is 2.
The SDK's retry policy also applies its own time budget. Use
`client.system_one(..., model=..., timeout=..., retry=...)` for individual requests.
Close the client when its owning application or operation ends.

Evaluation rubrics and thresholds belong to the service using Jev. The
[document grading service](document-grading.md) uses this factory to rank
retrieved chunks and produce a report. Authorize access before sending any
document content to the provider.
