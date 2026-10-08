# Simple Workflow

[English](README.md) | **简体中文**

一个强类型、极简的工作流引擎，由一套紧凑的 pipeline DSL 驱动。你用文本脚本描述一张工具调用
构成的 DAG；引擎解析它、**在执行前对整张图做类型检查**、剪掉所有到不了结果的节点，然后并发
执行剩下的节点。

```text
[bv:BV1TqaR67E7f] -> band-roformer[msst:Kim_MelBandRoformer.ckpt]
[var:band-roformer] -> vocals[unzip:vocals.wav]
...
[var:viridis-vocal, bands, harmony] -> result[audio:mix]
[var:result] -> [final]
```

## 特性

- **只有三种值类型**：`text`、`number`、`binary`。
- **静态类型检查**：逐条边检查类型，并检查工具参数的个数（arity）。
- **死代码消除**：到不了 `[final]`/`[output]` 的节点在执行前直接丢弃。
- **数据流 DAG 执行**：相互独立的分支并发运行；共享节点只执行一次。
- **两类工具**：
  - *API 工具*（`bv`、`msst`、`rvc`）——以 `POST` 表单调用外部 HTTP 服务。
  - *本地工具*（`unzip`、`audio`）——在进程内实现。
- 提供 HTTP API（FastAPI）与小型 CLI，使用 [uv](https://docs.astral.sh/uv/) 管理，
  并提供 Docker 镜像。

## 安装

```bash
uv sync                 # 创建虚拟环境并安装依赖
uv run pytest           # 运行测试
uv run simple-workflow serve
```

## 语言

### 值与类型

| 类型     | Python 表示 | 说明                              |
| -------- | ----------- | --------------------------------- |
| `text`   | `str`       |                                   |
| `number` | `float`     | 有限值                            |
| `binary` | `bytes`     | 流、zip、音频……                   |

值的类型会沿着每条边传递，并在编译期被检查。

### 元素

| 语法                            | 含义                                              |
| ------------------------------- | ------------------------------------------------- |
| `[tool:arg1,arg2]`              | 用字面量参数调用 `tool`（匿名节点）               |
| `name[tool:arg1]`               | 同上，并把输出绑定到变量 `name`                   |
| `[var:name]`                    | 引用此前绑定的变量                                |
| `[var:a, b, c]`                 | 把多个引用打包成一个多值输入                      |
| `[input]` / `[input:name]`      | 读取工作流输入（text/number/binary）              |
| `[final]` / `[output]`          | 终止汇点；其值即工作流结果                        |

`[var:a, b]` 与 `[var:a, var:b]` 等价 —— 每一项的 `var:` 前缀是可选的。

### 箭头 `->`

`A -> B` 把 `A` 的输出送入 `B`。具体来说，该值（若 `A` 是打包引用则为多个值）会填充
**`B` 在字面量参数之后剩余的参数**，也就是流入 `B` 的最后一个参数。

```text
[bv:ID] -> name[msst:model.ckpt]
#        \___ 字面量 "model.ckpt" 填充 a0，bv 的流填充 a1
```

链可以跨行；下一行可以以 `->` 开头，或当前行以 `->` 结尾。空行与 `#` 注释会被忽略。

### 示例

`examples/pipeline.workflow` 是完整的规范示例。它也可以写成紧凑形式（注意：RVC 这一步
必须命名为 `viridis-vocal`，因为后面会引用它）：

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

这里 `reverb-viridis-vocal` 从未被使用，会被静默剪掉。

## 工具

### API 工具

以 `POST` `multipart/form-data` 调用。text/number 值作为普通表单字段，binary 值作为文件
分片。默认字段名为 `a0`、`a1`……，但工具可以把它们映射为具体名字 —— 例如 MSST 服务期望
`model` 和 `audio`。响应体即返回值，按声明的输出类型解析（number → `float`，text → `str`，
binary → 原始字节）。

| 工具   | 签名                                     | 输出   | 请求                                        |
| ------ | ---------------------------------------- | ------ | ------------------------------------------- |
| `bv`   | `(bvid: text) -> binary`                 | binary | `GET .../video/{bvid}/download?kind=audio`  |
| `msst` | `(model: text, audio: binary) -> binary` | binary | `POST` 字段 `model`、`audio`（+ `response_format=zip`） |
| `rvc`  | `(model: text, audio: binary) -> binary` | binary | `POST` 字段 `model`、`audio`（+ `output_format=wav`） |

端点里可以包含 `{name}` 占位符（每个对应一个签名参数）；它们会被 URL 编码后的实参替换，
且该实参不会再次发送。每个端点是**完整 URL**，按如下顺序解析：

1. 运行请求里的 `tool_endpoints`，例如
   `{"msst": "http://host/api/msst/inference"}`；
2. 环境变量 `TOOLS_<NAME>_ENDPOINT`，例如
   `TOOLS_BV_ENDPOINT=http://xxx.xxx/api/b/video/{bvid}/download?kind=audio`。

可选的 API Key 可以通过请求里的 `tool_api_keys` 或环境变量 `TOOLS_<NAME>_API_KEY` 提供，
会以 `X-API-Key` 请求头发送。

`bv` 通过 bvid 下载 B 站音频流（`kind=audio` 查询参数与 `{bvid}` 路径占位符都在 URL 里）。

`msst` 对接 [MSST API](https://github.com/Deliay/msst-api)（`POST /api/msst/inference`）；
它返回一个包含各分轨的 zip（`vocals.wav`、`instrumental.wav`……），正好配合本地 `unzip` 工具。

`rvc` 对接 [Applio API 插件](https://github.com/Deliay/applio-api-plugin)（`POST /infer`）；
它直接返回转换后的音频（设置 `TOOLS_RVC_ENDPOINT=http://host:7870/infer`，若启用了鉴权再设
`TOOLS_RVC_API_KEY`）。

### 本地工具

| 工具    | 签名                                          | 输出   |
| ------- | --------------------------------------------- | ------ |
| `unzip` | `(path: text, archive: binary) -> binary`     | binary |
| `audio` | `(op: text, ...inputs: binary) -> binary`     | binary |

`audio` 支持的操作：

- `mono` —— 降混为单声道，
- `reverb` —— 施加 FFT 卷积混响，
- `mix` —— 把多路音频相加并做峰值归一化。

## HTTP API

启动服务：

```bash
TOOLS_BV_ENDPOINT='http://xxx.xxx/api/b/video/{bvid}/download?kind=audio' \
TOOLS_MSST_ENDPOINT=http://xxx.xxx/api/msst/inference \
TOOLS_RVC_ENDPOINT=http://xxx.xxx:7870/infer \
  uv run simple-workflow serve
```

| 方法   | 路径          | 说明                             |
| ------ | ------------- | -------------------------------- |
| `GET`  | `/health`     | 存活探针                         |
| `GET`  | `/v1/tools`   | 列出工具及其签名                 |
| `POST` | `/v1/compile` | 解析 + 类型检查，返回图          |
| `POST` | `/v1/run`     | 编译 + 执行                      |

`POST /v1/run`：

```jsonc
{
  "script": "...",
  "inputs": {
    "input": { "type": "binary", "data": "<base64>" }
  },
  "tool_endpoints": { "msst": "http://localhost:9000/api/msst/inference" },  // 可选覆盖
  "tool_api_keys": { "rvc": "secret" },                                      // 可选，X-API-Key
  "timeout": 300                                                             // 可选，秒
}
```

响应：

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

## 架构

```
src/simple_workflow/
  parser.py     # DSL -> 语句/元素
  compiler.py   # 建图、剪枝、拓扑排序、类型检查 -> Plan
  engine.py     # 并发 DAG 执行器
  registry.py   # 工具注册表
  signature.py  # 参数/返回值类型契约
  types.py      # text/number/binary + 转换与 JSON 编解码
  tools/
    api.py      # HTTP 后端工具
    local.py    # unzip + audio
  api.py        # FastAPI 应用
  __main__.py   # CLI
```

编译流程：

1. **Parse**：把脚本解析为元素。
2. **Build**：建立节点与 `->` 边；把 `name[...]` 的输出绑定进符号表。
3. **Resolve**：解析 `[var:x]` 引用（允许前向引用）。
4. **Prune**：从汇点反向可达性，剪掉到不了汇点的节点。
5. **Validate**：按拓扑序校验工具是否存在、参数个数、边的类型。
6. **Execute**：基于 `asyncio` 的依赖计数调度器执行。
