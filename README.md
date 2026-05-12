# Coven

**Multi-agent DAG pipeline framework.** Give it a task in plain English — it spawns specialized agents, builds a dependency graph, executes agents in parallel, and compiles a final output.

```bash
pip install coven
```

```python
from coven import Coven

coven = Coven(model="gpt-4o")
dag   = await coven.run("Produce a go-to-market strategy for a B2B SaaS product")
print(coven.to_text(dag))
```

---

## How it works

Coven runs five stages in sequence:

```
Task (plain English)
  │
  ▼
[1] Decomposer       — breaks task into focused agent nodes + artifacts
  │
  ▼
[2] Graph Builder    — validates wiring, formalizes edges, injects synthesizer nodes
  │
  ▼
[3] Sorter           — topological sort into parallel execution levels
  │
  ▼
[4] Executor         — runs each level concurrently via asyncio.gather
  │         └── per-node MCP server built automatically via ToolStorePy
  ▼
[5] Compiler         — assembles all artifact outputs into one coherent final result
```

Every stage is a separate agent with its own prompt, parser, and validator. No stage can corrupt the next.

### Artifacts

Artifacts are the edges of the DAG — structured JSON objects passed between agents. An artifact has contributors (agents that write to it) and users (agents that read from it). The graph structure emerges entirely from artifact wiring.

### Synthesizer injection

When multiple agents contribute to the same artifact, Coven automatically injects a Synthesizer node between them and the downstream consumers. It merges partial outputs, resolves conflicts, and attaches QC notes — without the decomposer needing to know this will happen.

### Parallel execution

Agents in the same topological level run concurrently. A pipeline with 10 agents across 3 dependency levels makes only 3 sequential round trips, not 10.

### Tool use via ToolStorePy

When the decomposer decides an agent needs external tools, it describes them in plain English inside `query_tool`. Before that agent executes, Coven calls [ToolStorePy](https://pypi.org/project/toolstorepy/) to semantically retrieve matching tool repositories, build a real MCP server, and make it available to the agent — automatically, per node, in parallel.

---

## Installation

**Requirements:** Python >= 3.12, an API key for any LiteLLM-supported model.

```bash
pip install coven
```

Copy `.env.example` to `.env` and fill in your API key:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

Coven uses [LiteLLM](https://github.com/BerriAI/litellm) — any supported provider works.

---

## Quick start

### CLI

```bash
uv run main.py "Produce a competitive analysis for a B2B SaaS product"
# or with a specific model
uv run main.py "Write a research report on LLM inference optimization" claude-sonnet-4-6
```

### Python

```python
import asyncio
from coven import Coven

async def main():
    coven = Coven(model="gpt-4o")
    dag   = await coven.run("Produce a go-to-market strategy for a B2B SaaS product")
    print(coven.to_text(dag))

asyncio.run(main())
```

### With a custom tool index

```python
coven = Coven(
    model="gpt-4o",
    mcp_index_url="https://example.com/my-tool-index.zip",
)
```

### Output

`coven.to_text(dag)` renders the compiled output:

```
============================================================
GO-TO-MARKET STRATEGY: B2B SAAS PRODUCT
============================================================

Executive summary of what was produced...

── Market Analysis ─────────────────────────────────────────
Market sizing, segments, and growth trends...
  [sources: market_analysis_report]

── Competitive Landscape ───────────────────────────────────
Competitor analysis and positioning gaps...
  [sources: competitive_analysis]

── Recommendations ─────────────────────────────────────────
  1. Target mid-market first — faster sales cycles
  2. Lead with integration story — buyers are already in the ecosystem
  ...

── Metadata ────────────────────────────────────────────────
  Agents: 6
  Artifacts: 8
============================================================
```

---

## Configuration

```python
Coven(
    model="gpt-4o",                    # Any LiteLLM-compatible model string
    workspace="coven_workspace",       # Root dir for artifacts and MCP servers
    mcp_index="core-tools",            # ToolStorePy built-in index (default)
    mcp_index_url=None,                # Custom index URL — overrides mcp_index
    mcp_install_requirements=False,    # Install tool repo requirements in venv
    mcp_verbose=False,                 # Verbose ToolStorePy logging
)
```

### Model strings

Any [LiteLLM model string](https://docs.litellm.ai/docs/providers) works:

```python
Coven(model="gpt-4o")
Coven(model="claude-sonnet-4-6")
Coven(model="azure/gpt-4o")
Coven(model="gemini/gemini-1.5-pro")
```

---

## Working with the DAG

The `dag` object returned by `coven.run()` contains the full execution record:

```python
dag = await coven.run("...")

# Final compiled output
dag.final_output          # dict: title, summary, sections, recommendations, metadata

# All nodes and their status
dag.nodes                 # dict[str, Node]
dag.nodes["agent_id"].status   # NodeStatus.COMPLETED | FAILED | PENDING

# All artifacts and their produced bodies
dag.artifacts             # dict[str, Artifact]
dag.artifacts["market_analysis_report"].body  # free-form JSON produced by the agent

# Execution structure
dag.levels                # list of lists — each inner list ran in parallel
dag.status                # DAGStatus.COMPLETED | FAILED
```

---

## Pipeline stages

### Decomposer

Takes the raw task and returns a list of domain agent nodes + artifacts. Each node gets:
- a scoped `system_prompt`
- `input_artifacts` it reads
- `output_artifacts` it produces
- `query_tool` entries describing any external tools it needs (plain English)

The decomposer prompt instructs the LLM to keep agents focused, avoid cycles, and use `query_tool` only when the agent genuinely needs external data or computation.

### Graph Builder

Takes the decomposed nodes and artifacts, verifies the wiring, derives explicit edges, repairs mismatches, detects cycles, and flags artifacts with multiple contributors for synthesizer injection. A hard algorithmic cycle check runs after the LLM stage.

### Sorter

Pure topological sort via networkx. Produces execution levels — each level is a list of node IDs that can run fully in parallel. No LLM involved.

### Executor

Runs each level with `asyncio.gather`. For each node with `query_tool` entries, it calls `MCPNodeBuilder` to build a dedicated ToolStorePy MCP server before execution. MCP builds within a level also run concurrently. Each node gets its own isolated workspace so parallel builds never conflict.

### Compiler

Reads all completed artifact bodies and compiles them into a single coherent output: title, summary, sections (each citing source artifacts), recommendations, and metadata.

---

## Project status

`v0.1.0` — early development. The pipeline architecture and API are stable. The following are not yet production-hardened:

- No retry logic for failed nodes
- No checkpoint/resume for long-running pipelines
- No streaming output during execution

Suitable for research workflows, prototyping, and experimentation.

---

## Dependencies

| Package | Purpose |
|---|---|
| `litellm` | Multi-provider LLM client |
| `instructor` | Structured LLM outputs via Pydantic |
| `networkx` | DAG construction and topological sort |
| `pydantic` | Data models throughout |
| `toolstorepy` | MCP server construction per node |
| `httpx` | Async HTTP |
| `python-dotenv` | Environment variable loading |

---

## Related projects

Coven is part of a broader open-source stack:

| Package | What it does |
|---|---|
| [toolstorepy](https://pypi.org/project/toolstorepy/) | Semantic MCP server builder — used internally by Coven |
| [sentinel-mlops](https://pypi.org/project/sentinel-mlops/) | Predictive observability layer for Prometheus + Grafana |
| [driftguard-ai](https://pypi.org/project/driftguard-ai/) | Semantic mistake memory and guardrails for autonomous agents |

---

## License

MIT