# Langfuse notes (read before writing the tracing / prompt / reader code)

Sources: the current Langfuse docs (read 2026-11-22: SDK overview, instrumentation, advanced features, token and cost tracking, prompt
management get-started / caching / guaranteed availability / link-to-traces, scores via SDK, Metrics API v2 + cookbook, pricing, billable units)
and the INSTALLED SDK (signatures and source inspected locally). Where the two could differ, the installed SDK is what this repo is tested against.

## Version

- **Python SDK v5, `langfuse==5.2.0`** (the newest version listed by the configured package index on 2026-11-22; it installed without
  changing any existing package; new dependencies: `backoff 2.2.1`, `opentelemetry-exporter-otlp-proto-http 1.45.2`,
  `opentelemetry-exporter-http-transport 0.66b2`, `wrapt 2.5.2`). Pinned in `requirements.txt` and `docs/env_baseline.txt`.
- v5 needs a Langfuse **server >= 4.59** (Langfuse Cloud qualifies; for a self-hosted v3 server use SDK v4). It is OpenTelemetry based and uses only the v4 public API.
- Upgrade paths exist (v3 to v4 to v5); this repo starts at v5, so nothing to migrate.

## Environment variables the SDK reads (v5.2.0, from `langfuse/_client/environment_variables.py`)

| Variable | Meaning |
|---|---|
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | project credentials (`pk-lf-...`, `sk-lf-...`) |
| **`LANGFUSE_BASE_URL`** | the host for this version (default `https://cloud.langfuse.com`, EU). `LANGFUSE_HOST` is still read as a legacy fallback. This project's `.env` uses `LANGFUSE_BASE_URL`. |
| `LANGFUSE_TRACING_ENABLED`, `LANGFUSE_SAMPLE_RATE`, `LANGFUSE_TIMEOUT`, `LANGFUSE_FLUSH_AT`, `LANGFUSE_FLUSH_INTERVAL`, `LANGFUSE_DEBUG`, `LANGFUSE_TRACING_ENVIRONMENT`, `LANGFUSE_RELEASE`, `LANGFUSE_PROMPT_CACHE_DEFAULT_TTL_SECONDS` | tuning |

`rag_core` does not rely on the SDK reading the environment on its own: `LangfuseTracer` / `LangfusePromptStore` build the client explicitly from the
variables (so tests can inject a fake and so "keys present" is decided by our code). `rag_core/env.py::load_env()` loads an ALLOWLIST of names from `.env`
(only entry-point scripts call it).

## Since 2026-11-27: traces are OpenTelemetry spans, not SDK observations

The sections below describe the SDK's own tracing API, which `rag_core` used until 2026-11-27. Traces now come from `rag_core/otel_tracing.py::OtelTracer`:
plain OpenTelemetry spans from a PRIVATE `TracerProvider`, exported with `OTLPSpanExporter` to `<base url>/api/public/otel/v1/traces`. The exporter sends
Basic auth with the key pair and `x-langfuse-ingestion-version: 4`, as the v5 SDK's exporter does; it is built in `rag_core/langfuse_client.py::langfuse_span_exporter`.
Each span carries the attributes the SDK would write (`langfuse.observation.type|input|output|level|status_message|metadata.<key>|model.name|model.parameters|
usage_details|cost_details|completion_start_time|prompt.name|prompt.version`, `langfuse.trace.name`, `session.id`), with the SDK's JSON encoding. The generation span
also carries the OpenTelemetry GenAI conventions (`gen_ai.*`). `tests/unit/test_rag_core_tracing.py::test_the_attributes_match_what_the_installed_langfuse_sdk_would_send`
compares the names and the encoding with the installed SDK (5.2.0). The trace id is the OTel trace id (32 hex characters), which is also the id that feedback scores use.
The SDK client is still used for prompts, scores and the Metrics API.

NOT verified live (the offline tests cannot show these; `scripts/smoke_trace.py` can, and it is run only when the owner asks):
- that Langfuse renders these spans exactly as it rendered the SDK's (observation types, prompt links, usage, sessions);
- that Langfuse does not count usage twice when both `langfuse.observation.usage_details` and `gen_ai.usage.*` are present (the docs say the `langfuse.*` attributes take precedence);
- the Metrics API delay: the "Data delay" note below says data from plain OTel exporters "can lag by up to 15 minutes". Whether the SDK-like headers avoid that is unknown, and the
  Observability tab may show new requests later than before.

## Creating traces and observations (Python v5)

- Client: `from langfuse import Langfuse, get_client, propagate_attributes`. `Langfuse(public_key=, secret_key=, base_url=, timeout=, sample_rate=, mask=, should_export_span=, flush_at=, flush_interval=, environment=, tracing_enabled=, debug=)`.
- **Context manager** (nesting is automatic through OpenTelemetry context): `with client.start_as_current_observation(as_type="span", name="rag.request", input=..., metadata=...) as span:` and children inside the block.
- **Manual**: `obs = client.start_observation(name=..., as_type=...)`; children via `obs.start_observation(...)`; you MUST call `obs.end()`; children of a "manual" observation must be created from that object, not from the client.
- `as_type` is one of `span, generation, embedding, agent, tool, chain, retriever, evaluator, guardrail`. Every observation counts as one billable unit.
- Update: `obs.update(output=..., metadata=..., level=..., status_message=...)`. The **trace** input/output are whatever the ROOT observation has; update the root to change them.
- Trace attributes (user id, session id, tags, metadata, environment, `prompt`) via `with propagate_attributes(session_id=..., user_id=..., tags=[...], metadata={...}):` around the root.
- `level` is one of `DEBUG | DEFAULT | WARNING | ERROR`, with `status_message`. Errors are recorded by setting `level="ERROR"` and a message; nothing is raised for us.
- IDs for downstream use: `client.get_current_trace_id()`, `obs.trace_id`, `obs.id`, `client.get_trace_url(trace_id=...)`.
- SDK errors are caught and logged by the SDK ("cannot break your application"); we still wrap every call ourselves (fail open) and count failures.

## Generation observations: usage, cost, completion_start_time

`start_as_current_observation(as_type="generation", name=, model=, model_parameters={...}, input=, output=, usage_details={...}, cost_details={...}, completion_start_time=<datetime>, prompt=<prompt client>, level=, status_message=)`.

- **Usage**: `usage_details={"input": n, "output": n}` (also accepts `input_tokens`/`output_tokens` in examples). Every key is a separate, NON-overlapping bucket; `total` is derived. Langfuse **no longer estimates tokens**: without submitted usage there is no usage and no cost.
  Reasoning tokens: if a provider reports them inside `completion_tokens` we record `output` as reported (inclusive) and do not add a second bucket, so nothing is double counted.
- **Cost**: either ingested (`cost_details={"input": usd, "output": usd}`, ingested cost takes priority) or INFERRED by Langfuse from usage + model name using its model definitions (prices per usage type; custom definitions can be added in the UI). We ingest cost only if `generation.prices` has the model id; we never invent a price. Langfuse may still infer a cost on its own if the model name matches one of its definitions.
- **`completion_start_time`**: a `datetime` for the time of the first token; Langfuse derives "time to first token" from it. Our value = request start + measured first-token latency.
- `model_parameters` values must be `str | int | float | bool | list[str] | None`.

## Scores

`client.create_score(name=, value=, trace_id=, observation_id=, session_id=, data_type="NUMERIC"|"CATEGORICAL"|"BOOLEAN"|"TEXT"|"CORRECTION", comment=, score_id=, config_id=, metadata=, timestamp=, environment=)`.
`BOOLEAN` takes `0`/`1`; `score_id` makes the call idempotent. A score is one billable unit. Scores are buffered like traces (flush).
Note (docs): on Langfuse Cloud `POST /api/public/ingestion` stops accepting everything except scores on 2026-11-16: scores still go through it; traces go through OpenTelemetry (v5 SDK).

## Flushing

The SDK exports in the background (batches). Short-lived scripts must call `client.flush()` (blocks until queued data is processed) or `client.shutdown()` (flush + stop threads; also registered with `atexit`). The Streamlit app is long running: it flushes on a bounded timeout after each request and shuts down at exit.

## Default span filtering (important)

v5 applies a default filter: a span is exported only if it was created by the Langfuse SDK (instrumentation scope `langfuse-sdk`), OR has a `gen_ai.*` attribute, OR comes from a known LLM instrumentation scope. Spans created by other OpenTelemetry libraries WITHOUT `gen_ai.*` attributes are dropped silently (and filtering can leave orphaned children). All our spans are created through the Langfuse client, so they pass. `should_export_span=` REPLACES the default (compose with `langfuse.span_filter.is_default_export_span`). Debug with `Langfuse(debug=True)`.
Other client options we use or know about: `sample_rate` (we also sample ourselves), `mask=` (a function applied to input/output/metadata; our masking happens BEFORE the SDK sees anything).

## Prompt management

- **Fetch**: `client.get_prompt(name, version=None, label=None, type="chat"|"text", cache_ttl_seconds=None, fallback=None, max_retries=None, fetch_timeout_seconds=None)`. Default label is `production`. A `fallback` (a list of chat messages for `type="chat"`) is returned with `prompt.is_fallback = True` if the fetch fails.
- **Caching**: client-side, in memory, stale-while-revalidate: a fresh entry is returned with no network; an expired entry is returned immediately and refreshed in the background; a cache MISS (first use in a process) fetches. `get_prompt` raises only if there is no cached copy and the network request fails after retries. That is why we pass `fallback=` AND wrap the call (belt and braces): answering must never block on Langfuse.
- **Create**: `client.create_prompt(name=, prompt=[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}], type="chat", labels=["production"], config=, commit_message=)` creates a NEW VERSION; versions are immutable; labels (`production`, `latest`, custom) move between versions: `client.update_prompt(name=, version=, new_labels=[...])`.
- **Variables**: `{{name}}` in message content. **Compile semantics** (read from `TemplateParser.compile_template`, verified by a test against the real function): ONE pass over the TEMPLATE; each `{{var}}` is replaced by `str(value)`; the inserted value is appended verbatim and is **never re-scanned**, so a `{{x}}`, `</source>` or "ignore previous instructions" inside a value stays inert; variables not passed are left as `{{var}}`; `None` becomes `""`. Chat prompts may also contain `placeholder` entries (message lists): our validator rejects any non-`message` entry.
- **Link a prompt to a generation**: pass `prompt=<prompt client>` to `start_as_current_observation(as_type="generation", ...)`; the generation header then shows "name - vN", and the prompt page lists its linked generations and per-version metrics. `propagate_attributes(prompt=...)` (SDK >= 4.14) links all generations in a context.
- **Free (Hobby) plan**: "Prompt versioning: Yes", "Prompt fetching: Unlimited", prompt caching, release management and playground are all included; no per-project cap on the number of prompts or versions is listed on the pricing page. Protected deployment labels are a paid add-on (so on Hobby anyone with project access can move the `production` label).
- **Do prompt fetches count toward usage?** No. Billable units are `traces + observations + scores` (docs: Billable Units). Creating a prompt version is not a unit either.

## Plan limits that shape the code (Hobby, from the pricing page)

- **50k units per month**, 30 days of data access, 2 users. Per request we emit one trace plus at most about 8 observations (+ one score on feedback): about 10 units, so roughly 160 requests a day would use the whole month at an even pace. `tracing.daily_request_cap` is therefore 150 and the span tree is kept small.
- **Rate limits**: general API 30 requests/minute; **Metrics API v2: 100 requests per DAY** on Hobby (100/hour on Core). The reader (`rag_core/observability_reader.py`) must cache aggressively: it uses one TTL cache (default 15 minutes), one query per panel, and returns a clear status instead of an error when it is rate limited.

## Reading data back: Metrics API v2

`GET {LANGFUSE_BASE_URL}/api/public/v2/metrics?query=<json>` with HTTP Basic auth (public key as user, secret key as password).
Query object: `{"view": "observations" | "scores-numeric" | "scores-categorical" | "scores-boolean", "metrics": [{"measure": "...", "aggregation": "count|sum|avg|p50|p75|p90|p95|p99|min|max"}], "dimensions": [{"field": "..."}], "filters": [{"column", "operator", "value", "type"}], "timeDimension": {"granularity": "hour|day|..."}, "fromTimestamp": iso, "toTimestamp": iso, "orderBy": [{"field": "sum_totalCost", "direction": "desc"}], "config": {"row_limit": 100}}`.
Result columns are named `{aggregation}_{measure}` (for example `sum_totalCost`, `p95_latency`, `count_count`) and `time_dimension`. Default row limit 100, max 1,000. The v2 API has no `traces` view (use `observations` and `isRootObservation` for logical roots). Measures seen in the docs: `count`, `latency`, `totalCost`, `value`; dimensions: `providedModelName`, `name`, `environment`, `userId`, `traceId`, `sessionId`. Measures and dimensions VERIFIED against the live API (scripts/smoke_trace.py, SDK 5.2.0, Cloud US region): measures `count`, `latency` (columns `p50_latency` / `p95_latency` / `p99_latency`), `timeToFirstToken` (`p50_timeToFirstToken`), `inputTokens` / `outputTokens` / `totalTokens` (`sum_inputTokens` ...), `totalCost` (`sum_totalCost`); the `scores-boolean` view with `value` (`avg_value`) and `count`. The API itself lists the valid observation dimensions in its HTTP 400 message: `id, evaluatorId, isEvaluatorTest, traceId, environment, parentObservationId, isRootObservation, type, name, level, version, userId, sessionId, aiGatewayKeyId, aiGatewayKeyOwnerUserId, tags, release, traceName, traceRelease, traceVersion, providedModelName, promptName, promptVersion, startTimeMonth, skillNamesAvailable, skillsNamesLoaded, skillIdsAvailable, skillIdsLoaded, loadedFilePaths, toolNames, calledToolNames, costType, usageType, experimentName, experimentDatasetId, experimentId`. **`statusMessage` is NOT a valid dimension or filter column** (HTTP 400): a first version of the reader filtered on it, the failed query was swallowed and the dashboard silently showed zero degraded requests. The root observation's `level` is now the signal instead: `ERROR` = failed request, `WARNING` = degraded (`RagService._close_trace`), and `tests/unit/test_rag_core_observability.py::test_every_dimension_and_filter_column_the_reader_sends_is_one_the_live_api_accepts` pins the list above. Filters that worked: `isRootObservation = true` (boolean), `type = GENERATION` (string), `name any of [...]` (stringOptions).
**Reading a single trace back**: in SDK v5 `client.api.trace` only has delete operations. Use `client.api.observations.get_many(trace_id=..., fields="core,basic,model,usage,prompt,metrics", limit=100)` (observations API v2, real-time like the Metrics API v2). Observations of one trace arrive piecemeal after a flush, so poll until the root is present and the count stops growing. Observed: generation `usage_details` come back with the keys `input`, `output` and `total` (Langfuse adds `total`); the prompt link comes back as `prompt_name` / `prompt_version`.
**Cost**: with `generation.prices` empty we send no cost, and the cost panel shows zero for the Groq model: Langfuse did not infer a price for that model id. A real price must be configured (per million tokens) before a cost figure means anything.
**Data delay**: data from SDK >= 4.7 appears in near real time on v2; older SDKs or plain OTel exporters can lag by up to 15 minutes. The reader reports its `as_of` time.

## Things that bit others (kept here so they do not bite us)

- Spans from non-Langfuse OpenTelemetry libraries without `gen_ai.*` attributes are filtered out by default.
- Forgetting `flush()` in a script loses the last traces.
- Forgetting `.end()` on a manual observation leaves it incomplete.
- Calling `get_prompt` without a `fallback` raises when Langfuse is unreachable and nothing is cached.
- A prompt edited in the Langfuse UI takes effect (after the cache TTL) without code review or tests: hence validation, a local fallback, version/hash on every trace, pinned versions for evaluation and a drift check (see PROJECT_CONTEXT).
