# AMS Studio Replica (React + FastAPI)

PRD-aligned ITIL ticket analytics platform that replicates the AMS Studio flow:
- Login
- Workspace list/create/edit
- Workspace settings (users, file storage, integrations)
- Ticket-X launch
- Data ingestion, mapping, SLA setup
- Dashboard + graph catalog + agentic query interface

All charts use **Plotly** as the single graphing library.

## Stack

- Frontend: React + Vite (`/Users/niranjansaxena/Desktop/AMS/frontend`)
- Backend: FastAPI (`/Users/niranjansaxena/Desktop/AMS/app.py`)
- Analytics engine: Pandas + Plotly (`/Users/niranjansaxena/Desktop/AMS/ticket_analytics`)

## Run (Single Command)

```bash
cd /Users/niranjansaxena/Desktop/AMS
npm run setup
npm run dev
```

This starts:
- Backend: `http://127.0.0.1:8010`
- Frontend: `http://127.0.0.1:5173` (or next available port)

## Alternative Run (Two Terminals)

Terminal 1:
```bash
cd /Users/niranjansaxena/Desktop/AMS
npm run backend
```

Terminal 2:
```bash
cd /Users/niranjansaxena/Desktop/AMS
npm run frontend
```

## Login Credentials

Use the same credentials you shared for AMS flow testing:
- Email: `niranjan.saxena@coforge.com`
- Password: `test123`

## Core Features Implemented

- Upload Excel/CSV ticket dumps.
- Mandatory mapping for:
  - Incident Number
  - Assignment Group
  - Application
- SLA mapping by priority (`P1`..`P4`).
- Robust preprocessing:
  - Alias normalization for common ITIL exports
  - Extended optional PRD fields (domain/sub-domain/subcategory/customer/environment/etc.)
  - MTTR fallback to `resolution_time` when timestamp delta is zero/missing
- Derived fields:
  - `category_derived`
  - `business_function_derived`
  - `mttr_hours`
  - `ticket_age_days`
  - `is_sla_breached`
  - `is_recurring_issue`
  - `team_performance_index`
- PRD dashboard card groups:
  - Incident Volumetrics
  - Delivery Compliance
  - Efficiency
  - Incident Composition
  - Performance
- PRD graph catalog (`Graph 1` to `Graph 8`) + insights.
- Agentic query modes:
  - Summarize Data
  - Auto Detect Issues
  - Enabler free-form query
- Multi-agent LLM query execution:
  - `IntentAgent` detects graph intent, metric, filters, and time window.
  - `GraphAgent` slices data, derives missing analytical columns when possible, and builds Plotly charts.
  - `ValidatorAgent` validates the generated output and returns explicit issues when data is unavailable.
- Custom graphs:
  - Word-cloud style Plotly view
  - Composite graph builder
- Exports:
  - Enriched CSV
  - Summary JSON
- Session resiliency:
  - Processed sessions are cached locally so dashboard/query calls survive backend restarts.

## Tests

Run backend tests:

```bash
cd /Users/niranjansaxena/Desktop/AMS
npm run test
```

Current status: all tests pass (`24 passed`).

## LLM Configuration

Set these env vars to enable LLM planner/validator agents:

```bash
export OPENAI_API_KEY="<your_api_key>"
export TICKETX_LLM_MODEL="gpt-4.1-mini"
```

If `OPENAI_API_KEY` is not set, deterministic fallback agents are used.

## Build Frontend

```bash
cd /Users/niranjansaxena/Desktop/AMS
npm run build
```

## Troubleshooting

- If `npm run dev` fails with `Address already in use`, another process is already running on port `8010` or `5173`.
- Stop existing local servers and rerun. The script now stops both processes automatically if backend startup fails.
- If you still see `Session not found`, reprocess the file once. The UI now auto-resets stale sessions and prompts reprocessing.
- Health check endpoint:

```bash
curl http://127.0.0.1:8010/api/health
```
