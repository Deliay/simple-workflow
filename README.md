# Simple Workflow

**English** | [简体中文](README.zh-CN.md)

A strongly-typed, minimal workflow engine driven by a compact pipeline DSL.
You describe a DAG of tool calls in a text script; the engine parses it,
**type-checks the graph before running**, prunes anything that does not reach a
result, and then executes the surviving nodes concurrently.

```text
[bv:BV1TqaR67E7f] -> band-roformer[msst:Kim_MelBandRoformer.ckpt]
[var:band-roformer] -> vocals[unzip:vocals.wav]
...
[var:viridis-vocal, bands, harmony] -> result[audio:mix]
[var:result] -> [final]
```

## Features

- **Three value types only**: `text`, `number`, `binary`.
- **Static type checking** of every edge, plus arity checks on tool parameters.
- **Dead-code elimination**: nodes that cannot reach `[final]`/`[output]` are
  dropped before execution.
- **Dataflow DAG execution** with independent branches running concurrently;
  shared nodes run exactly once.
- **Two tool classes**:
  - *API tools* (`bv`, `msst`, `rvc`) — `POST` form to an external HTTP service.
  - *Local tools* (`unzip`, `audio`) — implemented in-process.
- HTTP API (FastAPI) and a small CLI, packaged with [uv](https://docs.astral.sh/uv/)
  and shipped as a Docker image.

## Installation

```bash
uv sync                 # create the venv and install deps
uv run pytest           # run the test suite
uv run simple-workflow serve
```

## The language

### Values and types

| Type     | Python representation | Notes                                        |
| -------- | --------------------- | -------------------------------------------- |
| `text`   | `str`                 |                                              |
| `number` | `float`               | finite                                       |
| `binary` | `bytes`               | streams, zips, audio, ...                    |

A value's type travels with it along every edge and is checked at compile time.

### Elements

| Syntax                          | Meaning                                                   |
| ------------------------------- | --------------------------------------------------------- |
| `[tool:arg1,arg2]`              | call `tool` with literal arguments (an anonymous node)    |
| `name[tool:arg1]`               | same, and bind the output to variable `name`              |
| `[var:name]`                    | reference a previously bound variable                     |
| `[var:a, b, c]`                 | bundle several references into a multi-value input        |
| `[input]` / `[input:name]`      | read a workflow input (text/number/binary)                |
| `[final]` / `[output]`          | terminal sink; its value is a workflow result             |

`[var:a, b]` and `[var:a, var:b]` are equivalent — the `var:` prefix is optional
on the individual items.

### The arrow `->`

`A -> B` feeds the output of `A` into `B`. Concretely, the value (or values, if
`A` is a bundle) fill the **parameters of `B` that are left over after its
literal arguments**, i.e. the value flows into `B`'s last parameter.

```text
[bv:ID] -> name[msst:model.ckpt]
#        \___ literal "model.ckpt" fills param a0, the bv stream fills a1
```

Chains may span lines; a following line may start with `->`, or the current line
may end with `->`. Blank lines and `#` comments are ignored.

### Example

`examples/pipeline.workflow` contains the full specification example. It can be
written compactly (note: the RVC step must be named `viridis-vocal`, because it
is referenced later):

```text
[bv:BV1TqaR67E7f]
-> band-roformer[msst:Kim_MelBandRoformer.ckpt]
-> [unzip:vocals.wav]
-> karaoke-roformer[msst:bs_roformer_karaoke_frazer_becruily.ckpt]
-> [unzip:Vocals.wav]
-> [audio:mono]
-> [msst:dereverb_room_anvuew_sdr_13.7432.ckpt]
-> [unzip:noreverb]
-> viridis-vocal[rvc:viridis-v2_200e_5000s.pth]
-> reverb-viridis-vocal[audio:reverb]
[var:band-roformer] -> bands[unzip:other.wav]
[var:karaoke-roformer] -> harmony[unzip:Instrumental.wav]
[var:viridis-vocal, var:bands, var:harmony] -> result[audio:mix]
[var:result] -> [final]
```

Here `reverb-viridis-vocal` is never used, so it is pruned silently.

## Tools

### API tools

Invoked with `POST` `multipart/form-data`. Text/number values are plain form
fields, binary values are file parts. By default fields are named `a0`, `a1`, ...
but a tool can map them to concrete names — the MSST service, for example,
expects `model` and `audio`. The response body is the return value, parsed
according to the declared output type (number → `float`, text → `str`,
binary → raw bytes).

| Tool   | Signature                                | Output | Request                                  |
| ------ | ---------------------------------------- | ------ | ---------------------------------------- |
| `bv`   | `(bvid: text) -> binary`                 | binary | `GET .../video/{bvid}/download?kind=audio` |
| `msst` | `(model: text, audio: binary) -> binary` | binary | `POST` fields `model`, `audio` (+ `response_format=zip`) |
| `rvc`  | `(model: text, audio: binary) -> binary` | binary | `POST` fields `model`, `audio` (+ `output_format=wav`) |

The endpoint may contain `{name}` placeholders (one per signature parameter);
they are substituted with the percent-encoded argument, which is then not sent
again. Each endpoint is a **full URL**, resolved in this order:

1. the `tool_endpoints` field of a run request, e.g.
   `{"msst": "http://host/api/msst/inference"}`,
2. the `TOOLS_<NAME>_ENDPOINT` environment variable, e.g.
   `TOOLS_BV_ENDPOINT=http://xxx.xxx/api/b/video/{bvid}/download?kind=audio`.

An optional API key may be supplied via `tool_api_keys` on the request or the
`TOOLS_<NAME>_API_KEY` environment variable; it is sent as the `X-API-Key`
header.

`bv` downloads a Bilibili audio stream by bvid (the `kind=audio` query and the
`{bvid}` path placeholder are part of the URL).

`msst` targets the [MSST API](https://github.com/Deliay/msst-api)
(`POST /api/msst/inference`); it returns a zip of stems (`vocals.wav`,
`instrumental.wav`, ...), which pairs with the local `unzip` tool.

`rvc` targets the [Applio API plugin](https://github.com/Deliay/applio-api-plugin)
(`POST /infer`); it returns the converted audio directly (set
`TOOLS_RVC_ENDPOINT=http://host:7870/infer`, and `TOOLS_RVC_API_KEY` if you
enabled auth).

### Local tools

| Tool    | Signature                                    | Output |
| ------- | -------------------------------------------- | ------ |
| `unzip` | `(path: text, archive: binary) -> binary`     | binary |
| `audio` | `(op: text, ...inputs: binary) -> binary`     | binary |

`audio` operations:

- `mono` — down-mix to a single channel,
- `reverb` — apply an FFT-convolution reverb,
- `mix` — sum several streams and normalise peaks.

## HTTP API

Start the server:

```bash
TOOLS_BV_ENDPOINT='http://xxx.xxx/api/b/video/{bvid}/download?kind=audio' \
TOOLS_MSST_ENDPOINT=http://xxx.xxx/api/msst/inference \
TOOLS_RVC_ENDPOINT=http://xxx.xxx:7870/infer \
  uv run simple-workflow serve
```

| Method | Path          | Description                              |
| ------ | ------------- | ---------------------------------------- |
| `GET`  | `/health`     | liveness probe                           |
| `GET`  | `/v1/tools`   | list tools and their signatures          |
| `POST` | `/v1/compile` | parse + type-check, return the graph     |
| `POST` | `/v1/run`     | compile + execute                        |

`POST /v1/run`:

```jsonc
{
  "script": "...",
  "inputs": {
    "input": { "type": "binary", "data": "<base64>" }
  },
  "tool_endpoints": { "msst": "http://localhost:9000/api/msst/inference" },  // optional override
  "tool_api_keys": { "rvc": "secret" },                                      // optional, X-API-Key
  "timeout": 300                                                             // optional, seconds
}
```

Response:

```jsonc
{
  "outputs": { "final": { "type": "binary", "data": "<base64>" } },
  "elapsed": 1.23,
  "timings": [ { "node": "tool_4", "tool": "msst", "seconds": 0.9 } ]
}
```

## Docker

```bash
docker build -t simple-workflow .
docker run --rm -p 8000:8000 -e TOOLS_MSST_ENDPOINT=http://host.docker.internal:9000/api/msst/inference simple-workflow
```

## CLI

```bash
simple-workflow validate examples/pipeline.workflow
simple-workflow run examples/pipeline.workflow --binary input=track.wav --output-dir out/ \
  --endpoint msst=http://xxx.xxx/api/msst/inference
simple-workflow serve --port 8000
```

## Architecture

```
src/simple_workflow/
  parser.py     # DSL -> statements of elements
  compiler.py   # build graph, prune dead nodes, toposort, type-check -> Plan
  engine.py     # concurrent DAG executor
  registry.py   # tool registry
  signature.py  # param/output type contracts
  types.py      # text/number/binary + coercion & JSON codecs
  tools/
    api.py      # HTTP-backed tools
    local.py    # unzip + audio
  api.py        # FastAPI app
  __main__.py   # CLI
```

Compilation pipeline:

1. **Parse** the script into elements.
2. **Build** nodes and `->` edges; bind `name[...]` outputs into a symbol table.
3. **Resolve** `[var:x]` references (forward references allowed).
4. **Prune** nodes that cannot reach a sink (reverse reachability).
5. **Validate** in topological order: tool existence, parameter arity, and edge
   types.
6. **Execute** with a dependency-counting scheduler over `asyncio`.
