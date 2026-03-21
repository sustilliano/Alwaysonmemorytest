# always-on-memory

An always-on AI memory agent with **Rgano-inspired edge detection** and **trait extraction**, built in Rust.

Unlike flat memory systems that just store and retrieve, this agent understands the **topology** of your knowledge — where ideas connect, where they diverge, and where the interesting boundaries live.

---

## The Core Idea

```
edge_score(x) = 1 - consensus(x, neighbors(x))
```

This single formula runs through everything in this repo:

- In **memory space**: neighbors are memories sharing topics/entities. Low consensus = a genuine tension between related knowledge worth surfacing.
- In **image space** (`rgano_navigator.py`): neighbors are adjacent pixels. Low consensus = a visual edge — an obstacle, a dropoff, a wall.
- In **physical space**: sensor readings that disagree with expected = a boundary the robot needs to respond to.

Same math. Different domains.

---

## What's in here

| File/Dir | What it is |
|---|---|
| `src/` | Rust source for the always-on-memory daemon |
| `src/main.rs` | Entry point — spawns watcher, consolidation timer, HTTP server |
| `src/db.rs` | DuckDB memory store (memories, edges, consolidations tables) |
| `src/ingest.rs` | Inbox file watcher + LLM-powered content extraction |
| `src/edges.rs` | Rgano edge detection — computes `edge_score = 1 - consensus` across trait vectors |
| `src/consolidate.rs` | Timer-driven consolidation cycle — synthesizes insights from edge-adjacent memories |
| `src/query.rs` | Topology-aware query engine — returns context + detected tensions |
| `src/api.rs` | Native Axum HTTP API |
| `src/localrecall.rs` | LocalRecall-compatible API — drop-in replacement for LocalAGI's memory backend |
| `src/llm.rs` | LLM client (Ollama / LocalAI / exo / OpenAI-compat) |
| `src/config.rs` | TOML config structs |
| `src/revision.rs` | Revision detection utilities |
| `config.toml` | Configuration file — edit this to point at your LLM |
| `inbox/` | Drop `.txt` or `.md` files here — auto-ingested within seconds |
| `rgano_navigator.py` | Python: the same Rgano formula applied to robot vision/navigation |
| `Cargo.toml` | Rust dependencies |

---

## How it differs from Google's approach

| Google's approach | This approach |
|---|---|
| SQLite flat store | DuckDB columnar analytics |
| No dimensionality to memories | 16-axis trait vectors per memory |
| Consolidation = summarize | Consolidation = edge-aware synthesis |
| "What connections exist?" | "Where does consensus break down?" |
| Python daemon | Rust async daemon (Tokio) |
| Flash-Lite only | Any LLM (Ollama, OpenAI-compat) |

---

## Quick Start

```bash
# Clone and configure
cp config.toml my_config.toml
# Edit my_config.toml — set your LLM endpoint (Ollama, OpenAI, etc.)

# Build
cargo build --release

# Run (uses config.toml by default)
./target/release/always-on-memory

# Or with a custom config
./target/release/always-on-memory my_config.toml
```

### Drop files into the inbox

```bash
echo "AI agents need persistent memory to be useful" > inbox/note.txt
cp research_paper.md inbox/
# Auto-ingested within seconds
```

---

## HTTP API

### Native endpoints

```bash
# Ingest text directly
curl -X POST http://localhost:8888/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "DuckDB is an in-process analytical database", "source": "notes"}'

# Query with topology awareness
curl "http://localhost:8888/query?q=what+do+I+know+about+databases"

# Get memory stats
curl http://localhost:8888/status

# View detected knowledge edges (where consensus breaks down)
curl http://localhost:8888/edges

# List all memories
curl http://localhost:8888/memories

# List consolidation insights
curl http://localhost:8888/consolidations

# Trigger consolidation manually
curl -X POST http://localhost:8888/consolidate

# Delete a specific memory
curl -X POST http://localhost:8888/delete \
  -H "Content-Type: application/json" \
  -d '{"memory_id": 1}'

# Clear all memories
curl -X POST http://localhost:8888/clear
```

### LocalRecall-compatible endpoints (drop-in for LocalAGI)

```bash
# Create a collection
curl -X POST http://localhost:8888/api/collections \
  -H "Content-Type: application/json" \
  -d '{"name":"myCollection"}'

# Add content
curl -X POST http://localhost:8888/api/collections/myCollection/upload \
  -H "Content-Type: application/json" \
  -d '{"content":"AI agents need persistent memory", "source":"notes.txt"}'

# Search (topology-aware — returns tensions and insights, not just matches)
curl -X POST http://localhost:8888/api/collections/myCollection/search \
  -H "Content-Type: application/json" \
  -d '{"query":"what do I know about memory?", "max_results":5}'

# List entries
curl http://localhost:8888/api/collections/myCollection/entries

# Reset a collection
curl -X POST http://localhost:8888/api/collections/myCollection/reset
```

Point LocalAGI at this daemon instead of LocalRecall:

```bash
LOCALAGI_LOCALRAG_URL=http://always-on-memory:8888
```

---

## Configuration

Edit `config.toml`:

```toml
[server]
host = "0.0.0.0"
port = 8888

[watcher]
inbox_dir = "./inbox"
poll_interval_secs = 5

[consolidation]
interval_minutes = 30
min_unconsolidated = 3    # don't consolidate until at least this many memories
edge_threshold = 0.4      # minimum edge score to flag as an interesting boundary

[llm]
base_url = "http://localhost:11434"
model = "llama3.1:8b"
api_key = ""              # set for OpenAI-compatible cloud endpoints

[database]
path = "./memory.duckdb"

[traits]
dimensions = 16           # trait axes per memory (technical↔creative, cautious↔bold, etc.)
```

### LLM backend examples

```toml
# Ollama (local)
[llm]
base_url = "http://localhost:11434"
model = "llama3.1:8b"

# LocalAI (Docker)
[llm]
base_url = "http://localhost:8080"
model = "gemma-3-12b-it"

# exo (distributed cluster)
[llm]
base_url = "http://localhost:52415"
model = "mlx-community/Llama-3.2-1B-Instruct-4bit"

# OpenAI
[llm]
base_url = "https://api.openai.com"
model = "gpt-4o"
api_key = "sk-..."
```

---

## Architecture

```
                    ┌──────────────┐
                    │  inbox/      │  file watcher
                    │  *.txt *.md  │──────────┐
                    └──────────────┘           │
                                              ▼
  POST /ingest ─────────────────────► ┌──────────────┐
                                      │   Ingest     │
                                      │   Engine     │
                                      │  (LLM + trait│
                                      │   extraction)│
                                      └──────┬───────┘
                                             │
                                             ▼
                                      ┌──────────────┐
                                      │   DuckDB     │
                                      │  memories    │
                                      │  edges       │
                                      │  consolids   │
                                      └──────┬───────┘
                                             │
                          ┌──────────────────┼──────────────────┐
                          │                  │                  │
                          ▼                  ▼                  ▼
                   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
                   │  Rgano Edge │   │ Consolidate │   │    Query    │
                   │  Detection  │   │   (timer)   │   │   Engine    │
                   │             │◄──│             │   │  (topology  │
                   │ edge_score= │   │ LLM synth + │   │   aware)   │
                   │ 1-consensus │   │ edge-aware   │   │             │
                   └─────────────┘   └─────────────┘   └─────────────┘
```

**How consolidation works:**
1. Every N minutes (configurable), the consolidation timer fires
2. It grabs all unconsolidated memories and computes edge scores between them using their trait vectors
3. High-edge pairs (score > threshold) are memory boundaries — related but divergent
4. The LLM synthesizes an insight from each boundary cluster
5. Insights are stored as consolidations; source memories are marked consolidated

**How query works:**
1. Your query is embedded as a trait vector via the LLM
2. Memories are ranked by cosine similarity to your query vector
3. Top-K memories are pulled, plus any consolidation insights that reference them
4. The LLM synthesizes a response that includes not just matching content but detected tensions

---

## rgano_navigator.py

The same `edge_score = 1 - consensus(pixel, neighbors)` formula, applied to robot navigation.

Built for the **Freenove Tank Bot** — a tracked robot with a camera and ultrasonic sensor. Runs on the Pi onboard.

```python
from rgano_navigator import RganoNavigator
from car import Car
from camera import Camera

car = Car()
cam = Camera(stream_size=(320, 240))
cam.start_stream()

nav = RganoNavigator(car, cam)
nav.start()   # autonomous navigation loop in background thread
```

### Navigation regime cascade

The navigator runs a cascade of 5 regimes on every frame, then fuses their signals:

| Regime | What it does |
|---|---|
| `PathRegime` | Low-edge center region = clear corridor → FORWARD |
| `ObstacleRegime` | High-edge vertical bands = walls → steer away |
| `DropoffRegime` | Strong horizontal edge at floor level = cliff → REVERSE |
| `TargetRegime` | Edge-bounded cluster = object of interest → steer toward |
| `ConsensusRegime` | Fuses all signals + ultrasonic distance → final command |

The ultrasonic sensor acts as a hard override: under 15cm = emergency reverse regardless of vision.

### NavigationMemory

The robot can optionally feed its navigation decisions back into the always-on-memory daemon. Over time, the daemon consolidates spatial patterns — "this area always triggers obstacle+left signals" — giving the robot a topological memory of its environment.

```python
nav_mem = NavigationMemory("http://localhost:8888")
# call nav_mem.record(edge_map, decision, signals) each frame
```

---

## Stack

- **Rust 2024** + Tokio async runtime
- **Axum** HTTP framework
- **DuckDB** columnar memory storage
- **notify** filesystem watcher
- **tracing** structured logging
- Any **Ollama / LocalAI / exo / OpenAI-compatible** LLM
- **Python 3** + **NumPy** for the navigator
- Optional: **OpenCV** or **Pillow** for frame decoding in the navigator
