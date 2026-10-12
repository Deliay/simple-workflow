# Simple Workflow

**English** | [简体中文](README.zh-CN.md)

A strongly-typed, minimal workflow engine driven by a compact pipeline DSL.
You describe a DAG of tool calls in a text script; the engine parses it,
**type-checks the graph before running**, prunes anything that does not reach a
result, and then executes the surviving nodes (sequentially by default).

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
- **Dataflow DAG execution** with an optional concurrency limit (default 1, i.e.
  sequential); shared nodes run exactly once.
- **Layered S3 result cache**: every tool node's output is keyed by its own
  identity *and* the keys of the nodes feeding it, so a re-run reuses the
  longest unchanged prefix of the pipeline.
- **Two tool classes**:
  - *API tools* (`bv`, `msst`, `rvc`, `v2m`, `svs`) — `POST` form to an external HTTP service.
  - *Local tools* (`unzip`, `zip`, `audio`) — implemented in-process.
- HTTP API (FastAPI) and a small CLI, packaged with [uv](https://docs.astral.sh/uv/)
  and shipped as a Docker image.

## Installation

```bash
uv sync                 # create the venv and install deps
uv run pytest           # run the test suite
uv run simple-workflow serve
```

> **ffmpeg** must be on `PATH`. The `bv` tool returns an M4A stream, which the
> `msst`/`rvc` tools transcode to PCM WAV before calling their APIs. The Docker
> image already bundles it; for local runs install it with your package manager
> (e.g. `apt-get install ffmpeg`).

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
| `name[tool:arg]{key=value}`     | call `tool` and tag its value (see below)                 |
| `[!tool:arg]`                   | a leading `!` bypasses the node's cache (see below)       |

`[var:a, b]` and `[var:a, var:b]` are equivalent — the `var:` prefix is optional
on the individual items.

### Tags `{key=value}`

A `{...}` block after a value-producing node attaches metadata to that node's
output: `file=NAME` names it for `zip` and `mix=ROLE` assigns its role (e.g.
`vocal`) for `mix`. Tags are written at the node that produces the value, not on
the reference:

```text
a[bv:ID]{file=clip.mp4,mix=vocal} -> b[audio:amp]{file=loud.wav}
[var:a, var:b] -> [zip] -> [final]
```

### Inline bundles

Wrap several producers in an outer `[...]` — each item in its own brackets — to
feed them to the next node as one multi-value input. Items may carry their own
tags, and may be references or nested bundles:

```text
[[bv:1]{file=1.mp4}, [bv:2]{file=2.mp3}] -> [zip] -> [final]
```

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
-> viridis-vocal[rvc:viridis-v2_200e_5000s.pth]{mix=vocal}
-> reverb-viridis-vocal[audio:reverb]
[var:band-roformer] -> bands[unzip:other.wav]{mix=inst}
[var:karaoke-roformer] -> harmony[unzip:Instrumental.wav]{mix=harmony}
[var:viridis-vocal, var:bands, var:harmony] -> result[audio:mix]
[var:result] -> [final]
```

Here `reverb-viridis-vocal` is never used, so it is pruned silently, and the
`{mix=vocal}` tag marks the converted vocal as the lead track for `audio:mix`.

Two more compact examples using the MSST service's Vocal2Midi and DiffSinger
endpoints:

```text
# vocal audio -> zip of mid/ustx/txt artifacts (language forced, e.g. zh)
[input:vocal] -> [v2m:zh] -> [final]

# render an OpenUtau .ustx score with a voicebank -> wav
[input:score] -> rendered[svs:小松绿v1.1]
[var:rendered] -> [final]
```

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
| `v2m`  | `(language: text, audio: binary) -> binary` | binary | `POST` fields `language`, `audio` (+ `output_formats=mid,ustx,txt`, `output_pitch_curve=true`) |
| `svs`  | `(model: text, ustx: binary) -> binary`  | binary | `POST` fields `model`, `ustx` (+ `output_format=wav`) |

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

`v2m` targets the [MSST API](https://github.com/Deliay/msst-api)'s Vocal2Midi
endpoint (`POST /api/v2m/inference`); it turns vocal audio into a zip of
lyric-aligned artifacts (MIDI, USTX and plain text by default, with the pitch
curve enabled via `output_pitch_curve=true`), returned verbatim. The first
argument is a **required 2-letter language code**
(`zh`, `ja`, `en` or `yue`), sent as the `language` field so the service skips
its own detection — write `[v2m:zh]` under a `->` feeding the audio. Set
`TOOLS_V2M_ENDPOINT=http://host/api/v2m/inference`.

`svs` targets the same service's DiffSinger (singing voice synthesis) endpoint
(`POST /api/diffsinger/inference`); it renders an OpenUtau `.ustx` score with a
named voicebank such as `小松绿v1.1` and returns a WAV. Set
`TOOLS_SVS_ENDPOINT=http://host/api/diffsinger/inference`.

### Local tools

| Tool    | Signature                                    | Output |
| ------- | -------------------------------------------- | ------ |
| `unzip` | `(path: text, archive: binary) -> binary`     | binary |
| `zip`   | `(...files: binary) -> binary`                | binary |
| `audio` | `(op: text, ...inputs: binary) -> binary`     | binary |

`audio` operations:

- `mono` — down-mix to a single channel,
- `reverb` — apply a short, damped, dry-dominant FFT-convolution reverb tuned for pop material (pre-delay, high-frequency damping, rumble high-pass),
- `mix` — match every stream's loudness, then sum them. The `{mix=...}` tag picks a role (`vocal`/`harmony`/`inst`): the lead is lifted, the harmony sits below it, the bed is tucked down, and the lead side-chain ducks the rest (see below),
- `amp` — automatic gain: RMS-normalise a stream to a target loudness, with a peak ceiling as a fallback (silence untouched),
- `mp3` — encode the input to MP3 (ABR/VBR V0) through ffmpeg.

#### Vocal-forward mixing with `{mix=ROLE}`

Tag a track with its role: `{mix=vocal}` (lead), `{mix=harmony}` (support) or
`{mix=inst}` (instrumental bed). Anything untagged counts as the bed. `mix` lifts
the lead (`MIX_VOCAL_GAIN`), keeps the harmony below it (`MIX_HARMONY_GAIN`) and
tucks the bed down (`MIX_INST_GAIN`); the lead then **side-chain ducks** the other
roles — fully for the bed (`MIX_DUCK_DEPTH`) and more gently for the harmony
(`MIX_HARMONY_DUCK_DEPTH`). With no role tags, `mix` stays a plain equal-loudness
sum, so existing pipelines are unaffected.

```text
vocal[rvc:voice.pth]{mix=vocal}
harmony[unzip:Instrumental.wav]{mix=harmony}
bands[unzip:other.wav]{mix=inst}
[var:vocal, var:harmony, var:bands] -> [audio:mix] -> [final]
```

#### Naming bundled files with `{file=name}` and `[zip]`

`{file=name}` names the value a node produces. `[zip]` then packs every named
variable into a single archive, using those names as members. A `zip` input
without a `{file=...}` tag is an error.

```text
a[bv:test]{file=测试.mp4}
b[bv:test2]{file=测试2.mp4}
[var:a, var:b] -> [zip] -> [final]
```

produces a zip containing `测试.mp4` and `测试2.mp4`.

## HTTP API

Start the server:

```bash
TOOLS_BV_ENDPOINT='http://xxx.xxx/api/b/video/{bvid}/download?kind=audio' \
TOOLS_MSST_ENDPOINT=http://xxx.xxx/api/msst/inference \
TOOLS_RVC_ENDPOINT=http://xxx.xxx:7870/infer \
TOOLS_V2M_ENDPOINT=http://xxx.xxx/api/v2m/inference \
TOOLS_SVS_ENDPOINT=http://xxx.xxx/api/diffsinger/inference \
  uv run simple-workflow serve
```

| Method | Path          | Description                              |
| ------ | ------------- | ---------------------------------------- |
| `GET`  | `/health`     | liveness probe                           |
| `GET`  | `/v1/tools`   | list tools and their signatures          |
| `POST` | `/v1/compile` | parse + type-check, return the graph     |
| `POST` | `/v1/run`     | compile + execute                        |
| `DELETE` | `/v1/cache` | delete **every** cached node output      |

`POST /v1/run`:

```jsonc
{
  "script": "...",
  "inputs": {
    "input": { "type": "binary", "data": "<base64>" }
  },
  "tool_endpoints": { "msst": "http://localhost:9000/api/msst/inference" },  // optional override
  "tool_api_keys": { "rvc": "secret" },                                      // optional, X-API-Key
  "timeout": 600                                                             // optional per-node seconds; omit for no limit
}
```

Response:

```jsonc
{
  "outputs": { "final": { "type": "binary", "data": "<base64>" } },
  "elapsed": 1.23,
  "timings": [ { "node": "tool_4", "tool": "msst", "seconds": 0.9, "cached": false } ]
}
```

## Concurrency

Nodes execute through a bounded DAG scheduler. The default concurrency is **1**,
so the pipeline runs strictly sequentially and heavy tools never overlap. Raise
it with `SW_CONCURRENCY` to let independent branches run together:

```bash
SW_CONCURRENCY=4 uv run simple-workflow serve
```

## Timeouts

Each node's tool call runs **without a time limit** by default, so long-running
inference jobs are never cut off. Bound a single request's per-node timeout with
`timeout` (seconds), or set it for every run with `SW_TOOL_TIMEOUT`:

```jsonc
{ "timeout": 600 }  // per node, not for the whole run
```

```bash
SW_TOOL_TIMEOUT=600 uv run simple-workflow serve
```

An unset (or blank) `SW_TOOL_TIMEOUT` means no limit.

## Caching

Every tool node caches its output, and the cache is **layered**: a node's key is
a digest of its own identity (tool name + literal arguments) combined with the
keys of the nodes feeding it. A run therefore reuses the longest prefix of the
pipeline whose inputs are unchanged and only executes from the first node whose
inputs differ.

For the pipeline:

```text
[bv:BV1wM1vYsEn9]
-> band-roformer[msst:melband_roformer_instvox_duality_v2.ckpt]
-> [unzip:Vocals.wav]
-> [audio:mono]
-> [final]
```

the keys nest as follows:

| Node        | Cache key                                                |
| ----------- | -------------------------------------------------------- |
| `bv`        | `H("tool", "bv", "BV1wM1vYsEn9")`                        |
| `msst`      | `H("tool", "msst", "…duality_v2.ckpt", key(bv))`         |
| `unzip`     | `H("tool", "unzip", "Vocals.wav", key(msst))`            |
| `audio`     | `H("tool", "audio", "mono", key(unzip))`                 |

`[input]` nodes contribute a digest of the run-time value, so re-running a
workflow with different inputs invalidates exactly the affected subtree.
`DELETE /v1/cache` removes every entry.

### Bypassing the cache with `!`

Prefixing a node with `!` (e.g. `[!audio:mix]`) marks it as cache-bypassing: it
is never read from or written to the cache. A run-specific token is mixed into
its key, so **the node and everything downstream of it always re-execute** while
the unchanged prefix upstream is still reused. Use it after changing an
algorithm whose result would otherwise be served from a stale cache entry.

```text
[input:vocal] -> [audio:mono] -> [!audio:mix] -> [audio:mp3] -> [final]
#                                  \__________ re-runs every time __________/
```

Entries are stored in S3 and expire after a TTL (default **60 minutes**); a
background task periodically evicts stale objects. Enable the cache with:

```bash
# optional: point at MinIO / any S3-compatible endpoint
export SW_CACHE_ENDPOINT=http://minio:9000
export SW_CACHE_BUCKET=my-bucket
export SW_CACHE_PREFIX=simple-workflow/cache/   # optional
export SW_CACHE_TTL=3600                         # optional, seconds
uv run simple-workflow serve
```

Standard AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_REGION`, …) are picked up by boto3; the `SW_CACHE_ACCESS_KEY` /
`SW_CACHE_SECRET_KEY` / `SW_CACHE_REGION` / `SW_CACHE_SESSION_TOKEN` variables
override them. Without `SW_CACHE_BUCKET` (and without
`SW_CACHE_BACKEND=memory`) the cache is disabled. Other knobs:

| Variable                    | Default                  | Meaning                                   |
| --------------------------- | ------------------------ | ----------------------------------------- |
| `SW_CACHE_BACKEND`          | *(unset)*                | `memory` selects the in-process backend   |
| `SW_CACHE_CLEANUP_INTERVAL` | `300`                    | seconds between background cleanups       |

`DELETE /v1/cache` removes every entry and reports how many were deleted:

```jsonc
{ "enabled": true, "cleared": 42 }
```

## Docker

```bash
docker build -t simple-workflow .
docker run --rm -p 8000:8000 \
  -e TOOLS_MSST_ENDPOINT=http://host.docker.internal:9000/api/msst/inference \
  -e SW_CACHE_BUCKET=my-bucket \
  -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... \
  simple-workflow
```

The image installs the `cache` extra (`boto3`), so the S3 cache is ready to use
out of the box.

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
  engine.py     # bounded DAG executor (sequential by default)
  cache.py      # layered, S3-backed node result cache
  registry.py   # tool registry
  signature.py  # param/output type contracts
  types.py      # text/number/binary + coercion & JSON codecs
  tools/
    api.py      # HTTP-backed tools
    local.py    # unzip + zip + audio
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
