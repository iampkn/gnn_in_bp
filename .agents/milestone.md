# PROJECT MILESTONE TRACKER

## GNN Process Discovery (Khám phá Quy trình bằng Graph Neural Networks)

**Last Updated:** 2026-04-11 (v9)

---

## SOURCE MAP

```
Do_An/
├── .agents/
│   ├── project.md                          # Project specification (Vietnamese)
│   └── milestone.md                        # This file - progress tracker
├── paper/
│   └── 2109.05835v1.pdf                    # Reference paper (Sommers et al.)
├── docker-compose.yml                      # ✅ Docker Compose (FE + BE + Qdrant)
├── Dockerfile                              # ✅ Backend Docker image (Python 3.12)
├── .dockerignore                           # ✅ Backend Docker ignore rules
├── .env                                       # OPENAI_API_KEY (not committed)
├── requirements.txt                        # Python dependencies (DGL-free, pure PyTorch + openai)
├── fe/                                     # ✅ NextJS Frontend (Phase 5)
│   ├── Dockerfile                          # ✅ Frontend Docker image (Node 22, standalone)
│   ├── .dockerignore                       # ✅ Frontend Docker ignore rules
│   ├── next.config.ts                      # ✅ Standalone output for Docker
│   ├── app/
│   │   ├── page.tsx                        # ✅ Dashboard with tabbed UI + health check
│   │   ├── layout.tsx                      # Root layout with Geist fonts
│   │   ├── globals.css                     # Tailwind v4 + CSS vars
│   │   ├── lib/
│   │   │   └── api.ts                      # ✅ Typed API client (5 endpoints + health)
│   │   └── components/
│   │       ├── GenerateData.tsx            # ✅ Data generation form + results table
│   │       ├── SearchVector.tsx            # ✅ Vector search form + result cards
│   │       ├── DiscoverProcess.tsx         # ✅ Discovery form + Petri net display + XAI explanation
│   │       └── PetriNetViewer.tsx          # ✅ Interactive process template graph viewer (structural)
│   ├── package.json                        # Next 16.2.3, React 19.2.4
│   └── ...
├── src/
│   ├── __init__.py
│   ├── run_phase1.py                       # ✅ Phase 1 runner script
│   ├── data/
│   │   ├── csv/                            # Base data files
│   │   │   ├── node.csv                    # G1(14 nodes), G2(11 nodes)
│   │   │   ├── edge.csv                    # G1(15 edges), G2(11 edges)
│   │   │   ├── human.csv                   # 12 human resource assignments
│   │   │   └── device.csv                  # 17 device assignments
│   │   └── generated/                      # Generated output (Docker volume mount)
│   │       └── event_log_all.csv           # ✅ ~2,500 events, 220 cases, 176 activities
│   ├── phase1_data_generation/             # ✅ PHASE 1: COMPLETE
│   │   ├── __init__.py
│   │   ├── graph_reader.py                 # Read CSV -> ProcessGraph objects (G1, G2)
│   │   ├── graph_generator.py              # Orchestrate diverse template generation
│   │   ├── process_templates.py            # ✅ 20 unique IT process definitions (G3-G22)
│   │   └── event_log_simulator.py          # Simulate event logs with gateway branching
│   ├── phase2_vector_db/                   # ✅ PHASE 2: IMPLEMENTED
│   │   ├── __init__.py
│   │   ├── feature_extractor.py            # One-hot + freq + Cost + HumanRes vectors
│   │   └── vector_store.py                 # ✅ Qdrant client (env-var host) + in-memory fallback
│   ├── phase3_gnn/                         # ✅ PHASE 3: IMPLEMENTED (DGL-free)
│   │   ├── __init__.py
│   │   ├── graph_encoder.py                # Encode L -> G (trace graph + candidate Petri net)
│   │   │                                   #   AlphaRelations, candidate place generation
│   │   │                                   #   HeteroGraphData (pure-Python, replaces DGL)
│   │   ├── propagation_net.py              # PN1 (4-layer GCN+attention) & PN2 (2-layer)
│   │   │                                   #   Eq.(1)(2): bi-directional, K-head attention, RELU
│   │   ├── select_candidate.py             # SCN: single FC layer + SOFTMAX (Eq.3,4)
│   │   │                                   #   S-coverability constraint integration
│   │   ├── stop_network.py                 # SN: 2-layer with gating SIGMOID (Eq.5,6)
│   │   │                                   #   Workflow net override check
│   │   ├── s_coverability.py               # S-coverability & workflow net checker
│   │   ├── discovery_model.py              # Full model: d = (f, PN1, SCN, SN, PN2)
│   │   │                                   #   forward_train (teacher forcing)
│   │   │                                   #   forward_inference (sequential selection)
│   │   ├── training.py                     # Training loop: NLL loss (Eq.7), BFS ordering
│   │   │                                   #   Adam optimizer, gradient clipping
│   │   └── inference.py                    # Beam search inference (width b=10/50)
│   │                                       #   Returns ranked Petri net candidates
│   ├── phase4_regression/                  # ✅ PHASE 4: IMPLEMENTED
│   │   ├── __init__.py
│   │   └── regression_head.py              # Cost/HumanRes prediction from graph embedding
│   └── backend/                            # ✅ PHASE 5 BACKEND: IMPLEMENTED
│       ├── __init__.py
│       ├── main.py                         # ✅ FastAPI app with env-var CORS for Docker
│       └── routes/
│           ├── __init__.py
│           ├── generate_data.py            # POST /api/generate-data
│           ├── search_vector.py            # GET  /api/search-vector
│           ├── discover_process.py         # POST /api/discover-process
│           ├── petri_net.py               # GET  /api/petri-net (template graph structure)
│           └── explain_net.py            # ✅ POST /api/explain-net (XAI via OpenAI)
```

---

## PHASE STATUS

### Phase 1: Data Generation & Preprocessing ✅ COMPLETE

- **Status:** Fully implemented and tested
- **Output:** 22 graphs (2 base + 20 unique templates), ~2,500 events, 220 traces (10/graph), 176 unique activities
- **Key files:** `graph_reader.py`, `graph_generator.py`, `process_templates.py`, `event_log_simulator.py`
- **Run:** `python src/run_phase1.py`
- **Details:**
    - G1 (IT project lifecycle, 14 nodes) and G2 (incident management, 11 nodes) from CSV
    - G3-G22: **20 unique IT process templates** defined in `process_templates.py`, each with:
      - Different activities, topology, gateway patterns, and flow logic
      - Covering: CI/CD, network ops, IAM, change mgmt, backup, security,
        procurement, release mgmt, bug tracking, DB migration, cloud infra,
        pentest, onboarding, disaster recovery, API integration,
        performance optimization, agile sprint, IT audit, microservices, ETL
    - Node count varies from 8-13 per process; gateway count 1-2 per process
    - Simulates 10 traces per graph (default, configurable) with ExclusiveGateway branching (loop probability decay)
    - Output saved as CSV event logs

### Phase 2: Vector Database Integration ✅ IMPLEMENTED

- **Status:** Feature extractor and vector store implemented
- **Key files:** `feature_extractor.py`, `vector_store.py`
- **Details:**
    - Feature vectors: one-hot activity label + frequency + cost + human_res
    - Qdrant client with automatic fallback to in-memory numpy cosine search
    - ✅ `VectorStore` reads `QDRANT_HOST` / `QDRANT_PORT` env vars for Docker
    - ✅ Qdrant deployed via Docker Compose (`qdrant/qdrant:latest`)
    - Accessible via `/api/search-vector` endpoint

### Phase 3: GNN Core Architecture ✅ IMPLEMENTED

- **Status:** All 4 neural networks implemented per paper specifications
- **Architecture (from paper Section IV):**
    - **PN1:** 4-layer GCN with K-head attention → Eq.(1)(2)
    - **SCN:** Single FC layer + SOFTMAX → Eq.(3)(4) with S-coverability
    - **SN:** 2-layer with SIGMOID gating → Eq.(5)(6) with workflow net check
    - **PN2:** 2-layer GCN with K-head attention
- **Graph Encoding:** Alpha-relations for candidate place pruning (Appendix A)
- **Graph Data Structure:** `HeteroGraphData` — lightweight pure-Python heterogeneous graph
    - ✅ Replaced DGL dependency (was causing `torchdata.datapipes`, `libcurl.so.4`,
      `graphbolt` version mismatch errors in Docker)
    - Implements same interface: `ntypes`, `canonical_etypes`, `num_nodes()`,
      `nodes[ntype].data`, `edges(etype=...)`
    - All GNN layers (PN1/PN2) are pure PyTorch — DGL was only used as a data container
- **Training:** NLL loss (Eq.7), teacher forcing with BFS ordering
- **Inference:** Beam search (configurable width)
- **TODO:** Train model on generated data, evaluate with conformance metrics

### Phase 4: Regression Head ✅ IMPLEMENTED

- **Status:** Cost/HumanRes prediction layer implemented
- **Key file:** `regression_head.py`
- **TODO:** Train alongside discovery model, evaluate prediction accuracy

### Phase 5: Web Interface ✅ IMPLEMENTED

- **Backend:** FastAPI with 5 API routes
    - `POST /api/generate-data` → Run Phase 1 pipeline
    - `GET  /api/search-vector` → Query similar tasks via Qdrant
    - `POST /api/discover-process` → GNN inference → Petri net JSON
    - `GET  /api/petri-net?graph_id=G1` → Return process template graph structure (each node once)
    - `POST /api/explain-net` → **XAI**: send template graph + discovered net to OpenAI for business explanation
    - ✅ CORS origins configurable via `CORS_ORIGINS` env var
    - ✅ `OPENAI_API_KEY` from `.env` file, passed via `env_file` in docker-compose
- **Frontend:** NextJS 16 interactive dashboard in `fe/`
    - ✅ Tabbed dashboard UI (4 tabs) with backend health check indicator
    - ✅ **Generate Data** tab: configure variants/traces/seed, view summary stats + per-graph table
    - ✅ **Vector Search** tab: search by activity/graph_id/min_cost, view scored result cards
    - ✅ **Process Discovery** tab: select graph_id or upload CSV, view Petri net structure
        - ✅ **XAI "Ask AI" button** per Petri Net card: calls OpenAI (gpt-4o-mini) to explain
          the discovered net in Vietnamese for managers — includes meaning, comparison with
          template, optimization suggestions, bottleneck warnings
    - ✅ **Petri Net** tab: interactive process template graph viewer
        - Shows **template graph structure directly** — each node appears exactly once
        - Gateway nodes visible with branching edges (all branches shown)
        - BFS-layered layout: circles (Start/End), rectangles (Activity), diamonds (Gateway)
        - Clean arrows without inflated frequency labels
        - Zoom (native wheel listener, `passive: false`) + Pan (drag) controls
        - Click any node for details (activity, type, cost, human_res)
        - Summary bar: node count, edge count, total cost
        - Legend with node type explanation (Start, Activity, End, Gateway, Flow)
    - ✅ Typed API client (`fe/app/lib/api.ts`) with full error handling
    - ✅ Dark mode support via Tailwind CSS
- **TODO:**
    - Display cost predictions from regression head

### Phase 6: Docker Deployment ✅ IMPLEMENTED & VERIFIED

- **Status:** Full Docker Compose stack with 3 services, all endpoints tested
- **Services:**
    | Service    | Image                    | Port | Description                        |
    | ---------- | ------------------------ | ---- | ---------------------------------- |
    | `qdrant`   | `qdrant/qdrant:latest`   | 6333 | Vector database (persistent volume)|
    | `backend`  | Python 3.12-slim (built) | 8000 | FastAPI backend                    |
    | `frontend` | Node 22-alpine (built)   | 3000 | NextJS standalone production build |
- **Key files:**
    - `docker-compose.yml` — orchestrates all 3 services
    - `Dockerfile` — backend image (pip install + src copy)
    - `fe/Dockerfile` — frontend multi-stage build (deps → build → standalone runner)
    - `.dockerignore` / `fe/.dockerignore` — optimized build contexts
- **Usage:**
    ```bash
    # Start all services
    docker compose up --build

    # Access
    #   Frontend:  http://localhost:3000
    #   Backend:   http://localhost:8000
    #   Qdrant UI: http://localhost:6333/dashboard

    # Stop
    docker compose down

    # Remove volumes (clean Qdrant data)
    docker compose down -v
    ```
- **Environment variables:**
    | Variable             | Service  | Default                                      |
    | -------------------- | -------- | -------------------------------------------- |
    | `QDRANT_HOST`        | backend  | `qdrant` (Docker) / `localhost` (local dev)  |
    | `QDRANT_PORT`        | backend  | `6333`                                       |
    | `CORS_ORIGINS`       | backend  | `http://localhost:3000,http://localhost:3001` |
    | `NEXT_PUBLIC_API_URL`| frontend | `http://localhost:8000` (build arg)           |
- **Volumes:**
    - `qdrant_data` — persistent Qdrant storage
    - `./src/data` → `/app/src/data` — shared data directory (CSV + generated)

---

## CHANGE LOG

### 2026-04-11 — Added XAI (Explainable AI) feature to Process Discovery tab

**Problem:** The Process Discovery tab showed technical metrics (log-probability, place
structure, transitions) that are meaningful for data scientists but difficult for managers
to interpret. Managers couldn't understand how to apply these results to optimize workflows.

**Solution:** Added an "Ask AI to Explain" button after each discovered Petri Net card.
When clicked, it sends the **template graph structure** (nodes, edges, costs, human_res)
plus the **discovered net** (rank, log_probability, transitions, places) to OpenAI's
GPT-4o-mini model. The AI responds in Vietnamese with a structured business analysis:

1. **Ý nghĩa** — What the discovered Petri Net means in business context
2. **So sánh** — How it compares to the original template process
3. **Đề xuất tối ưu** — Suggestions to shorten/optimize the process, highlight costly steps
4. **Rủi ro** — Bottleneck warnings and gateway monitoring recommendations

**Implementation:**
- New backend route: `POST /api/explain-net` in `src/backend/routes/explain_net.py`
  - Receives `graph_id` + `discovered_net` (rank, log_prob, transitions, places)
  - Loads the template graph (G1-G22) for full context
  - Builds a detailed Vietnamese prompt with both template and discovered net data
  - Calls OpenAI `gpt-4o-mini` (temperature=0.7, max_tokens=1500)
  - Returns `{ explanation: string }` in Vietnamese
- Frontend: "Ask AI to Explain" button in `DiscoverProcess.tsx`
  - Amber-themed button with lightbulb icon
  - Loading spinner during API call
  - Explanation rendered as **Markdown** via `react-markdown` + `@tailwindcss/typography`
    (headings, bold, bullet lists, numbered lists all display correctly)
  - "Hide explanation" toggle to collapse
- `OPENAI_API_KEY` read from `.env` file, passed to Docker via `env_file` in docker-compose

**Files changed:**
- `src/backend/routes/explain_net.py` — **new file**: XAI endpoint
- `src/backend/main.py` — registered `explain_net` router
- `requirements.txt` — added `openai>=1.0.0`
- `docker-compose.yml` — added `env_file: .env` to backend service
- `fe/app/lib/api.ts` — added `ExplainNetRequest`, `ExplainNetResult`, `explainNet()`
- `fe/app/components/DiscoverProcess.tsx` — added "Ask AI" button + markdown explanation panel
- `fe/app/globals.css` — added `@plugin "@tailwindcss/typography"` for prose styles
- `fe/package.json` — added `react-markdown`, `@tailwindcss/typography` dependencies

---

### 2026-04-10 — Fixed inflated event counts: show template graph structure, reduce traces

**Problem:** The Petri Net viewer was mining a DFG from 200 traces per graph, causing
every step to show 200 occurrences (e.g., "Bat_Dau" → occurrence=200). In a real
workflow, each step happens once per execution. Total events were 49,773 — unrealistically high.

**Root cause:** Two issues combined:
1. `traces_per_graph` default was 200, inflating all event counts
2. The `/api/petri-net` endpoint mined a DFG (aggregating 200 traces), losing the
   clean structural view — every node showed occurrence=200, every edge showed frequency=200

**Fix:**
1. Rewrote `/api/petri-net` to return the **process template graph directly** instead
   of mining a DFG from event logs. Each node now appears exactly once with its
   designed cost. Gateway nodes (ExclusiveGateway) are visible with ALL outgoing
   branches shown — no information is lost from random trace sampling.
2. Reduced default `traces_per_graph` from 200 → 10 (still configurable via API param).
   This brings total events from ~49,773 to ~2,514 for 22 graphs.
3. Updated frontend: removed edge frequency labels (all structural edges = 1), removed
   frequency-proportional line thickness, cleaned node detail panel to show Cost and
   Human Res instead of "Occurrences", added Gateway to the legend.

**Files changed:**
- `src/backend/routes/petri_net.py` — rewrote: reads template graphs directly, no event log needed
- `src/backend/routes/generate_data.py` — default `traces_per_graph` 200 → 10
- `fe/app/components/PetriNetViewer.tsx` — clean structural view, no frequency labels

**Result:** G10 now shows 13 nodes (each once), 14 edges, total_cost=840, with visible
gateway branching — matching the designed process template exactly.

---

### 2026-04-10 — Replaced repeated graph variants with 20 unique IT process templates

**Problem:** `GraphGenerator` only varied cost/humanres on the SAME 2 base topologies.
G3-G22 were copies of G1/G2 with different numbers — all 22 graphs had identical flow
patterns (just 2 unique processes repeated 11 times each).

**Fix:** Created `process_templates.py` with 20 genuinely unique IT process templates,
each with different activities, topology, gateway patterns, and flow logic:

| Graph | Process | Activities | Gateways |
|-------|---------|-----------|----------|
| G3 | CI/CD Pipeline | Commit, Build, Unit Test, Integration Test, Deploy | 2 (test gates + loop) |
| G4 | Network Troubleshooting | Cảnh báo, Kiểm tra kết nối, Phân tích log | 1 (hardware vs software) |
| G5 | User Access Management | Xác minh, Phê duyệt, Cấu hình quyền | 1 (accept/reject) |
| G6 | Change Management (ITIL) | RFC, Đánh giá rủi ro, Họp CAB, Rollback | 2 (risk level + verify) |
| G7 | Backup & Recovery | Sao lưu, Kiểm tra toàn vẹn, Lưu trữ offsite | 1 (integrity loop) |
| G8 | Security Incident Response | Phát hiện, Forensic, Khắc phục lỗ hổng | 1 (severity level) |
| G9 | Hardware Procurement | Duyệt ngân sách, Đặt hàng, Kiểm tra chất lượng | 2 (budget + quality) |
| G10 | Software Release | Feature freeze, Regression test, UAT, Deploy | 2 (regression + UAT) |
| G11 | Bug Tracking | Báo cáo lỗi, Sửa lỗi, Code review, QA | 2 (review + QA loops) |
| G12 | Database Migration | Schema, Script, Test dev, Migrate, Rollback | 2 (test + data verify) |
| G13 | Cloud Infrastructure | IaC Terraform, Apply, Network, Security | 1 (health check loop) |
| G14 | Penetration Testing | Scan, Khai thác, Leo thang quyền | 1 (vulnerability found?) |
| G15 | IT Onboarding | Email, Quyền, VPN, Đào tạo công cụ | 1 (training loop) |
| G16 | Disaster Recovery | Đánh giá thiệt hại, DR site, Khôi phục | 2 (severity + verify) |
| G17 | API Integration | Phân tích spec, Code, Test API, Deploy gateway | 1 (test loop) |
| G18 | Performance Optimization | Monitoring, Bottleneck, Tối ưu, Load test | 2 (found? + improved?) |
| G19 | Agile Sprint | Planning, User Story, Standup, Review, Retro | 1 (code review loop) |
| G20 | IT Audit | Phạm vi, Bằng chứng, Phỏng vấn, Báo cáo | 1 (compliant vs violation) |
| G21 | Microservices Development | API Contract, Docker, K8s, Service Mesh | 2 (unit test + mesh) |
| G22 | ETL Data Pipeline | Extract, Transform, Validate, Load DW, BI | 1 (validation loop) |

**Result:** 22 graphs with **178 unique activities**, 49,773 total events.
Each graph produces a visually distinct DFG in the Petri Net viewer.

**Files changed:**
- `src/phase1_data_generation/process_templates.py` — **new file**: 20 unique process definitions
- `src/phase1_data_generation/graph_generator.py` — rewritten to use diverse templates
- Old event log deleted → user must click "Generate Data" to regenerate

---

### 2026-04-10 — Rewrote Petri Net Viewer: DFG mining from event logs + scroll-zoom fix

**Bug 1: Scroll zoom not working — browser page scrolled instead**
- The `useEffect` for the native wheel listener had `[]` dependency, meaning it ran
  on mount when `loading=true` and the SVG wasn't rendered yet (`svgRef.current === null`)
- Fix: added `data` to the dependency array so the listener attaches after data loads
  and the SVG is actually in the DOM. Also added `e.stopPropagation()`.

**Bug 2: All graphs showed identical topology (only cost differences)**
- The old endpoint read static template graphs from `node.csv`/`edge.csv` and used
  `GraphGenerator` to create variants — but variants only differ in cost/humanres,
  the topology (nodes + edges) was always identical to G1 or G2.
- Fix: completely rewrote `/api/petri-net` to **mine a Directly-Follows Graph (DFG)**
  from `event_log_all.csv`. Now:
  - Reads event log, filters by `graph_id`
  - Extracts traces per case and builds a DFG: activities = nodes, sequential pairs = edges
  - Each edge carries a `frequency` count showing how many times that transition occurred
  - Node `cost` is the average cost per occurrence, `occurrence` shows total event count
  - Each graph shows genuinely different patterns because gateway choices vary per trace
- Frontend updated: edge thickness proportional to frequency, frequency labels on edges,
  summary shows trace count / activity count / flow count

**Example frequency differences (all from same base topology G1):**
- G1: Coding→Kiem_Thu_UAT=479, Rollback=163, total_cost=663,650
- G3: Coding→Kiem_Thu_UAT=471, Rollback=155, total_cost=700,779
- G5: Coding→Kiem_Thu_UAT=473, Rollback=164, total_cost=671,066

**Files changed:**
- `src/backend/routes/petri_net.py` — complete rewrite: DFG mining from event logs
- `fe/app/components/PetriNetViewer.tsx` — edge frequency display, scroll-zoom fix
- `fe/app/lib/api.ts` — added `frequency` to `PetriNetEdge`, `case_count` to result

---

### 2026-04-10 — Added interactive Petri Net Viewer tab

**Feature:** New "Petri Net" tab in the dashboard for visualizing process graphs.

**Backend:** `GET /api/petri-net?graph_id=G1` endpoint in `petri_net.py`
- Reads graph structure from `node.csv` / `edge.csv` via `GraphReader`
- Computes BFS-layered layout (x,y positions for each node)
- Returns nodes (with positions, types, costs) + edges + available graph IDs

**Frontend:** `PetriNetViewer.tsx` — interactive SVG component
- Graph selection dropdown (auto-populates from backend)
- Proper Petri net notation: circles (Start/End), rounded rectangles (Task), diamonds (Gateway)
- Directed edges with arrowheads, edge routing from/to node borders
- Zoom (scroll wheel), pan (click-drag), node click for detail panel
- Summary bar with node count, edge count, total cost
- Legend explaining node type colors

**Files added/changed:**
- `src/backend/routes/petri_net.py` — new endpoint
- `src/backend/main.py` — registered new router
- `fe/app/components/PetriNetViewer.tsx` — new visualization component
- `fe/app/lib/api.ts` — added `getPetriNet()` + types
- `fe/app/page.tsx` — added "Petri Net" tab
- `.agents/project.md` — updated Phase 5 spec

---

### 2026-04-10 — Removed DGL dependency, fixed Docker runtime errors

**Problem:** `POST /api/discover-process` returned 500 errors inside Docker:
1. `No module named 'torchdata.datapipes'` — DGL 2.1.0 (PyPI) imported deprecated torchdata module
2. `libcurl.so.4: cannot open shared object file` — DGL runtime dependency missing in slim image
3. `Cannot find DGL C++ graphbolt library ...pytorch_2.11.0.so` — DGL 2.1.0 incompatible with torch 2.11.0

**Root cause:** DGL on PyPI is stuck at v2.1.0, which has hard dependencies on
`torchdata.datapipes` (removed in torchdata ≥0.9), `libcurl`, and torch-version-specific
C++ graphbolt `.so` files. These create cascading failures with modern PyTorch (≥2.4).

**Fix:** Replaced DGL entirely with `HeteroGraphData` — a lightweight pure-Python class
in `graph_encoder.py` that provides the same interface (`ntypes`, `edges()`, `num_nodes()`,
`nodes[ntype].data`). This was safe because:
- All GNN layers (PN1, PN2, SCN, SN) were already pure PyTorch
- DGL was only used as a data container via `dgl.heterograph()`
- Removed unused `import dgl` / `from dgl.nn.pytorch import GATConv` in `propagation_net.py`

**Files changed:**
- `src/phase3_gnn/graph_encoder.py` — added `HeteroGraphData` class, removed `import dgl`
- `src/phase3_gnn/propagation_net.py` — removed unused `import dgl` and `GATConv` import
- `src/phase3_gnn/__init__.py` — exported `HeteroGraphData`
- `requirements.txt` — removed `dgl`, `torchdata`, `pyyaml` (DGL deps)
- `Dockerfile` — removed `DGLBACKEND=pytorch` env var

**Verified:** `POST /api/discover-process?graph_id=G1&beam_width=10` returns 200 OK with
5 ranked Petri net candidates (12 activities, 153 candidate places, 200 traces).

---

## NEXT STEPS (Priority Order)

1. **Train GNN model** on generated event logs with synthetic Petri net pairs
2. ~~**Add Petri net visualization**~~ ✅ Done — interactive SVG viewer with zoom/pan/click
3. **Evaluate** model with conformance metrics (fitness, precision, simplicity)
4. **Integration testing** of full pipeline (data → GNN → Petri net → UI)

---

## DATA SUMMARY

| Metric               | Value                                                  |
| -------------------- | ------------------------------------------------------ |
| Base graphs (CSV)    | 2 (G1: IT lifecycle, G2: Incident mgmt)                |
| Template processes   | 20 unique IT workflows (G3-G22)                        |
| Total graphs         | 22 (all with distinct topology and activities)         |
| Traces per graph     | 10 (default, configurable via API)                     |
| Total traces/cases   | 220                                                    |
| Total events         | ~2,514                                                 |
| Unique activities    | 176                                                    |
| Avg events per trace | ~11.4                                                  |
| IT domains covered   | CI/CD, Security, Network, Cloud, Data, Agile, Audit... |

---

## PAPER REFERENCE

- **Title:** Process Discovery Using Graph Neural Networks
- **Authors:** Sommers, Menkovski, Fahland (TU Eindhoven)
- **Key contribution:** Supervised learning for APD using GCN with attention
- **Architecture:** PN1 → SCN → SN → PN2 loop with beam search inference
- **Training:** Teacher forcing with BFS ordering, NLL loss
- **Constraints:** S-coverability for soundness, workflow net check for stopping
