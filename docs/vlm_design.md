# VLM 粗标注与 L3 质检设计

## 0. 范围

这份设计覆盖两件事：

1. **粗标注 `annotate`**：从原始 egocentric 视频生成 episode 描述、粗粒度子任务，或两者都生成。
2. **L3 质检 `vlm_audit`**：把 VLM 审计从 L1/L2 中拆出来，独立失败、独立重试、独立限流，最后写一个 `final.json` 决策。

不在第一版范围内：

- 帧级精确的时序分割；
- 逐子任务调用 VLM；
- `Step.is_done` 的内容指纹校验；
- 任何需要把 API key、数据集、模型权重或输出提交进仓库的流程。

---

## 1. 为什么 `annotate` 是并行分支，不是独立管线

原 DAG：

```text
retarget ──► base_ik ──► approach ──► render ─┐
segment ──► inpaint ──────────────────────────┼──► composite ──► curate
depth ────────────────────────────────────────┘
```

加入粗标注后：

```text
annotate ─────────────────────────────────────────────┐
retarget ──► base_ik ──► approach ──► render ─┐       │
segment ──► inpaint ──────────────────────────┼──► composite ──► curate ──► vlm_audit
depth ────────────────────────────────────────┘
```

`annotate`：

- 读 `src.hdf5` 和 `src.mp4`；
- 不依赖 retarget / segment / depth / render / composite；
- 从 t=0 就能和 GPU 阶段并行；
- 结果写到 `annotation/coarse.json`；
- 只被 `vlm_audit` 消费，不阻塞主合成链。

它是同一个 batch、同一个 `Job`、同一个调度器里的一个 stage，不是另一条 pipeline。  
把它塞进 `curate` 会让所有标注等待 composite 完成，并让 `curate` 重新依赖外部 API，和拆分 L3 的目标冲突。

如果只想离线预跑，也可以：

```bash
python -m egowhale episode.hdf5 --steps annotate --annotate-mode both
```

后续 batch 里的 `annotate` 看到输出已存在会跳过。

---

## 2. 为什么没有独立的 `finalize` stage

最终决策只有三步：

1. 读 `action/quality.json`；
2. 读 `action/vlm_audit.json`；
3. 写 `action/final.json`。

它没有模型、没有 GPU、没有网络，单独做成 DAG stage 只会多一个调度单元和一个依赖。  
因此 `finalize.py` 只保留纯函数：

```python
def decide(quality: dict, audit: dict, fail_open: bool = True) -> dict:
    ...
```

`VlmAudit.run()` 在写完 audit 后直接调用它，并同时写：

```text
action/vlm_audit.json
action/final.json
```

如果以后需要在不重新调用 API 的情况下重算最终决策：

```bash
python -m egowhale.quality.finalize <out_dir>
```

这样 DAG 少一个节点，同时保留“final 可独立重算”的能力。

---

## 3. 数据契约

### 3.1 `annotation/coarse.json`

```json
{
  "status": "ok",
  "meta": {
    "mode": "both",
    "prompt_version": "coarse_v1",
    "model": "qwen-vl-plus",
    "frames": 420,
    "fps": 30.0,
    "description_source": "hdf5",
    "sampled_frames": 24,
    "max_subtasks": 8,
    "min_subtask_frames": 15
  },
  "episode_summary": "Pick up the red block and place it into the box.",
  "subtasks": [
    {"start_frame": 0, "end_frame": 120, "instruction": "Reach for the red block"},
    {"start_frame": 120, "end_frame": 260, "instruction": "Grasp and lift the block"},
    {"start_frame": 260, "end_frame": 420, "instruction": "Place it into the box"}
  ],
  "error": null
}
```

约定：

- `status` 为 `ok | skipped | error | disabled`；
- `start_frame` 包含，`end_frame` 不包含；
- 子任务帧号相对原始视频；
- `mode` 为 `off | description | subtasks | both`；
- `description_source` 为 `hdf5 | vlm | mixed | none | default`。

三种模式：

| 模式 | 已有 HDF5 描述 | 行为 |
| --- | --- | --- |
| `off` | 任意 | 不调用 API，写 `disabled` |
| `description` | 有 | 直接复用，不调用 API |
| `description` | 无 | 调用 VLM 生成描述 |
| `subtasks` | 有/无 | 调用 VLM 生成子任务；已有描述作为上下文 |
| `both` | 有 | 复用描述，只调用 VLM 生成子任务 |
| `both` | 无 | 一次调用生成描述和子任务 |

子任务规整：

- 支持 `start_frame` / `end_frame`，也支持秒数；
- clamp 到 `[0, frames]`；
- 排序；
- gap 由前一段补到后一段的起点；
- overlap 以后一段 `start_frame = 前一段 end_frame` 截断；
- 过短段合并；
- 超过 `max_subtasks` 时合并最短的相邻段；
- 第一段从 0 开始，最后一段到 `frames`。

### 3.2 `action/quality.json`

只含 L1/L2：

```json
{
  "keep_l1_l2": true,
  "keep": true,
  "frames": 420,
  "fps": 30.0,
  "approach": 12,
  "description": "Pick up the red block...",
  "description_source": "hdf5",
  "invalid_ratio": 0.12,
  "mean_ik_error_m": 0.004,
  "hand": [],
  "ik": [],
  "visible": [],
  "self_collision": [],
  "cross_arm": [],
  "too_large": [],
  "l1": [],
  "outlier": [],
  "sudden": [],
  "valid": []
}
```

- `keep_l1_l2` 是主字段；
- `keep` 是兼容旧脚本的别名；
- 不再写 `vlm` 字段；
- `frames`、`fps`、`approach` 给 `vlm_audit` 复用，避免重复读文件。

### 3.3 `action/vlm_audit.json`

```json
{
  "status": "ok",
  "model": "qwen-vl-plus",
  "prompt_version": "audit_v1",
  "scope": "episode",
  "is_consistent": true,
  "confidence": 0.86,
  "reasoning": "The robot picks up the block and places it into the box.",
  "mismatches": [],
  "description": "Pick up the red block...",
  "error": null
}
```

- `status` 为 `ok | skipped | error | disabled`；
- `is_consistent` 为 `true | false | null`；
- 第一版 `scope=episode`，子任务只作为 prompt 上下文；
- `null` 表示模型没有给出明确结论；
- 所有 status 都带 `error` 字段（无错误时为 `null`），skipped/disabled 额外带 `reason`。

### 3.3.1 审计参照的选择

`vlm_audit` 检查的是视频和"任务描述"是否一致，描述按以下优先级取：

1. `quality.json` 中 `description_source` 为 `hdf5`/`mixed` 的描述（ground truth）；
2. `annotation/coarse.json` 的 `episode_summary`（无标注数据时由 VLM 生成）；
3. 重新读 HDF5 属性，最后退到 `"manipulation"`。

也就是说 HDF5 里的原始描述优先于 VLM 自己写的 summary，避免"同一模型生成描述再自我验证"的循环。子任务列表无论哪种来源都作为 prompt 上下文带上。

### 3.4 `action/final.json`

```json
{
  "keep": true,
  "keep_l1_l2": true,
  "vlm_status": "ok",
  "vlm_is_consistent": true,
  "decision": "keep",
  "reason": "l1_l2 and vlm agree"
}
```

`decision` 枚举：

| 条件 | decision |
| --- | --- |
| L1/L2 不通过 | `drop_l1_l2` |
| L1/L2 通过，VLM 明确 `false` | `drop_vlm` |
| 两者都不通过 | `drop_both` |
| L1/L2 通过，VLM `true` 或 `null` | `keep` |
| L1/L2 通过，VLM 不可用，`fail_open=true` | `keep_vlm_unavailable` |
| L1/L2 通过，VLM 不可用，`fail_open=false` | `drop_vlm_unavailable` |
| audit 被禁用 | `keep_audit_disabled` |

---

## 4. 配置

```yaml
vlm:
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  model: qwen-vl-plus        # 按百炼控制台实际模型 ID 替换
  api_key_env: EGOWHALE_VLM_API_KEY
  timeout_s: 120
  retries: 2
  backoff_s: 2.0
  cache: true

annotate:
  mode: off                  # batch 默认关闭，避免无意产生费用
  sample_fps: 1.0
  max_frames: 24
  max_subtasks: 8
  min_subtask_frames: 15
  prompt_version: coarse_v1
  on_error: keep             # keep | fail

audit:
  enabled: true
  sample_fps: 4.0
  max_frames: 96
  prompt_version: audit_v1
  fail_open: true
  scope: episode

nodes:
  vlm:
    actors: 4

io:
  cpus:
    vlm: 1
```

API key 只从 `vlm.api_key_env` 指向的环境变量读取，不写进 YAML，也不写进输出 JSON。

---

## 5. VLM 层

### 5.1 客户端

`egowhale/vlm/client.py` 提供：

- `VLMClient`：`complete(messages)` / `complete_json(messages)`；
- `OpenAICompatibleClient`：标准库 `urllib`，兼容 OpenAI chat completions；
- `MockClient`：测试和本地 mock 用，按队列返回字符串或抛异常。

### 5.2 重试

| 情况 | 行为 |
| --- | --- |
| 401 / 403 | 不重试，`error.kind=auth` |
| 400 | 不重试，`error.kind=http` |
| 429 | 重试，指数退避 |
| 5xx | 重试，指数退避 |
| 超时 / 连接失败 | 重试，指数退避 |
| JSON 解析失败 | 一次 repair 机会；再失败写 `error` |

默认 `retries=2`，退避 `2s, 4s`，带少量 jitter。

JSON 解析失败时，`request_json` 会带上 `repair_prompt`（期望的 schema 来自 `prompts.py` 的 `*_SCHEMA` 常量）再问一次；repair 的结果同样进缓存。再失败写 `error`。

### 5.3 缓存

- 缓存目录：`<episode_out>/vlm_cache/`；
- key：

```text
sha256(model + prompt_version + messages)
```

- `messages` 包含图片 data URL，所以图片内容变更会使缓存失效；
- 只缓存成功解析的 JSON；
- 不缓存 401/403；
- 不缓存原图，不写 key。

### 5.4 限流

每个 `VlmActor` 进程内同步调用，总并发由：

```yaml
nodes:
  vlm:
    actors: 4
```

控制。没有引入 Redis 等外部限流组件。

注意：百炼免费档的 QPS 很低。429 会按指数退避重试，但如果持续打满重试上限，episode 会变成 `status=error`（`fail_open=true` 下是 `keep_vlm_unavailable`）。首次真机跑建议 `actors` 从 1–2 起步，确认 429 不再频繁后再调大。

---

## 6. 批处理接入

- `nodes.vlm.actors` 新增一个不占 GPU 的 actor pool；
- `VlmActor` 持有 `Annotate` 和 `VlmAudit`；
- `CpuActor` 只跑 `composite` / `curate`；
- 删除旧的 `CpuActor` 中 `os.environ.pop("EGOWHALE_VLM_API_KEY")`；
- `batch/report.py` 的 boot 表显示 VLM pool；
- `resources.py` 把 VLM 计入 CPU worker 预算；
- `batch.jsonl` 每行带 `decision`（episode 写出 `final.json` 时）；
- 整批结束时打一行 `decisions: keep N, drop_l1_l2 M, ...` 汇总，L3 的效果不需要逐个 grep `final.json`。

`vlm` pool 和 GPU 阶段解耦：

- GPU 忙时，`annotate` 可以并行；
- `vlm_audit` 等 `curate` 完成后才跑；
- 没有 key 时两个 stage 都快速写 `skipped`，不会拖住 GPU 链。

---

## 7. 失败语义

### 7.1 `annotate`

- 没有 key / 模型名为空：`status=skipped`；
- API 超时 / 429 / 5xx：重试后 `status=error`；
- 解析失败：`status=error`；
- `annotate.on_error=keep`：写错误，不抛异常，不阻断 episode；
- `annotate.on_error=fail`：抛异常，episode 失败。

### 7.2 `vlm_audit`

- `keep_l1_l2=false`：`status=skipped`，不调用 API；
- `audit.enabled=false`：`status=disabled`；
- 没有 key / 模型名为空：`status=skipped`；
- API 错误：`status=error`；
- composite 解码失败：`status=error`（不炸 episode，由 `fail_open` 决定去留）；
- 明确 `is_consistent=false`：`status=ok`，`is_consistent=false`；
- 明确 `true`：`status=ok`，`is_consistent=true`；
- 没有结论：`status=ok`，`is_consistent=null`。

### 7.2.1 旧版 `quality.json`

`frames`/`fps`/`approach` 是这次才写进 `quality.json` 的。旧输出目录续跑时，`vlm_audit` 会从
`action/prefix.npz` 和 composite 本身重建这三个值，因此旧 episode 也会得到真正的审计，而不是
静默变成 `keep_vlm_unavailable`。

### 7.3 `fail_open`

`fail_open` 不是“允许失败”，而是“质检本身不可用时默认放行”：

- `fail_open=true`：只有明确 `is_consistent=false` 才丢；
- `fail_open=false`：audit 不可用（`skipped/error`）也丢。

无论哪种，`status`、`reason`、`error` 都会写进 JSON，不会静默吞掉。

第一版默认 `fail_open=true`，避免 API 故障把整批数据丢掉。

---

## 8. 文件存在性与版本

第一版**不做** `Step.is_done` 的内容指纹校验。

现有 `Step` 只检查 `makes` 文件是否存在，因此：

- 换 `annotate.mode`、`prompt_version`、`model` 时，建议使用新的 `out` 目录；
- 或者在同一个 `out` 里手动删除 `annotation/coarse.json`；
- 文档和脚本不承诺自动失效。

后续可以加：

```python
def is_done(self, dst, index=None) -> bool:
    ...
```

用 `meta.mode`、`meta.prompt_version`、`meta.model`、`meta.max_frames` 等字段判断，并让 batch 的 `Job` 持有配置后的 step 实例。

---

## 9. 测试

不需要 GPU、不需要真实 API key：

- `tests/test_vlm_client.py`：MockClient、429 重试、401 不重试、缓存、repair 轮次；
- `tests/test_vlm_wire.py`：本地起一个 OpenAI 兼容的假端点，让真实 `OpenAICompatibleClient` 走真 HTTP，
  验证请求体结构（messages/role/content/data URL/Authorization）、429 重试、repair 轮次，以及
  annotate 和 vlm_audit 两条 stage 的端到端路径；不需要外网和真实 key；
- `tests/test_vlm_schema.py`：JSON 提取、verdict 解析、子任务 gap/overlap/clamp/merge；
- `tests/test_annotate.py`：三种模式、复用 HDF5 描述、错误策略、解码失败不炸 episode；
- `tests/test_vlm_audit.py`：true/false/null、skipped、error、fail-open/fail-closed、decide 矩阵、
  旧版 quality.json 的几何回填、HDF5 描述优先于 VLM summary。

真实 API 只作为手工验证：

```bash
export EGOWHALE_VLM_API_KEY=...
export EGOWHALE_VLM_MODEL=<百炼模型 ID>
python -m egowhale episode.hdf5 --steps annotate --annotate-mode both
```

---

## 10. 安全与数据

- API key 只从环境变量读；
- 不把 key、图片、视频、模型权重、数据集、输出 JSON 提交进仓库；
- `outputs/`、`thirdparty/`、`datasets/` 已在 `.gitignore`；
- 缓存只写 `<out>/vlm_cache/`，该目录在 `outputs/` 下；
- 提示词不包含公司内部数据；
- 日志只记录状态、模型、耗时，不记录 base64 图片或 key。
