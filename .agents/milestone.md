# PROJECT MILESTONE TRACKER

## GNN Process Discovery (Khám phá Quy trình bằng Graph Neural Networks)

**Last Updated:** 2026-04-10 (v5)

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
├── requirements.txt                        # Python dependencies (DGL-free, pure PyTorch)
├── fe/                                     # ✅ NextJS Frontend (Phase 5)
│   ├── Dockerfile                          # ✅ Frontend Docker image (Node 22, standalone)
│   ├── .dockerignore                       # ✅ Frontend Docker ignore rules
│   ├── next.config.ts                      # ✅ Standalone output for Docker
│   ├── app/
│   │   ├── page.tsx                        # ✅ Dashboard with tabbed UI + health check
│   │   ├── layout.tsx                      # Root layout with Geist fonts
│   │   ├── globals.css                     # Tailwind v4 + CSS vars
│   │   ├── lib/
│   │   │   └── api.ts                      # ✅ Typed API client (all 4 endpoints + health)
│   │   └── components/
│   │       ├── GenerateData.tsx            # ✅ Data generation form + results table
│   │       ├── SearchVector.tsx            # ✅ Vector search form + result cards
│   │       ├── DiscoverProcess.tsx         # ✅ Discovery form + Petri net display
│   │       └── PetriNetViewer.tsx          # ✅ Interactive DFG visualization (mined from event logs)
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
│   │       ├── event_log_all.csv           # ✅ 64,408 events, 4,400 cases
│   │       └── event_log_G{N}.csv          # ✅ Per-graph event logs (22 files)
│   ├── phase1_data_generation/             # ✅ PHASE 1: COMPLETE
│   │   ├── __init__.py
│   │   ├── graph_reader.py                 # Read CSV -> ProcessGraph objects
│   │   ├── graph_generator.py              # Generate 20 IT process variants
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
│           └── petri_net.py               # GET  /api/petri-net (DFG mined from event logs)
```

---

## PHASE STATUS

### Phase 1: Data Generation & Preprocessing ✅ COMPLETE

- **Status:** Fully implemented and tested
- **Output:** 22 graphs (2 base + 20 variants), 64,408 events across 4,400 traces
- **Key files:** `graph_reader.py`, `graph_generator.py`, `event_log_simulator.py`
- **Run:** `python src/run_phase1.py`
- **Details:**
    - Reads G1 (IT project lifecycle, 14 nodes) and G2 (incident management, 11 nodes)
    - Generates 20 variants (G3-G22) with varied Cost (±60-150%), HumanRes, Roles, Devices
    - Simulates 200 traces per graph with ExclusiveGateway branching (loop probability decay)
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

- **Backend:** FastAPI with 4 API routes
    - `POST /api/generate-data` → Run Phase 1 pipeline
    - `GET  /api/search-vector` → Query similar tasks via Qdrant
    - `POST /api/discover-process` → GNN inference → Petri net JSON
    - `GET  /api/petri-net?graph_id=G1` → Mine DFG from event logs with edge frequencies
    - ✅ CORS origins configurable via `CORS_ORIGINS` env var
- **Frontend:** NextJS 16 interactive dashboard in `fe/`
    - ✅ Tabbed dashboard UI (4 tabs) with backend health check indicator
    - ✅ **Generate Data** tab: configure variants/traces/seed, view summary stats + per-graph table
    - ✅ **Vector Search** tab: search by activity/graph_id/min_cost, view scored result cards
    - ✅ **Process Discovery** tab: select graph_id or upload CSV, view Petri net structure
    - ✅ **Petri Net** tab: interactive Directly-Follows Graph (DFG) visualization
        - Mines process model from event log traces (not static templates)
        - Each graph shows unique execution frequencies and flow patterns
        - Edge labels show transition frequency, line thickness proportional to frequency
        - BFS-layered layout: circles (Start/End), rectangles (Activity)
        - Zoom (native wheel listener, `passive: false`) + Pan (drag) controls
        - Click any node for details (activity, type, avg cost, occurrences)
        - Summary bar: trace count, activity count, flow count, total cost
        - Legend with node type + frequency explanation
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

| Metric               | Value                                   |
| -------------------- | --------------------------------------- |
| Base graphs          | 2 (G1: IT lifecycle, G2: Incident mgmt) |
| Generated variants   | 20 (G3-G22)                             |
| Total graphs         | 22                                      |
| Traces per graph     | 200                                     |
| Total traces/cases   | 4,400                                   |
| Total events         | 64,408                                  |
| Unique activities    | 20                                      |
| Avg events per trace | ~14.6                                   |

---

## PAPER REFERENCE

- **Title:** Process Discovery Using Graph Neural Networks
- **Authors:** Sommers, Menkovski, Fahland (TU Eindhoven)
- **Key contribution:** Supervised learning for APD using GCN with attention
- **Architecture:** PN1 → SCN → SN → PN2 loop with beam search inference
- **Training:** Teacher forcing with BFS ordering, NLL loss
- **Constraints:** S-coverability for soundness, workflow net check for stopping
