# 🎧 Qwen3-TTS 有声书生成器

基于阿里 Qwen3-TTS 的开源语音合成工具，支持音色克隆、长文本分段合成、WebUI 交互式编辑与 CLI 批量处理。

> 核心引擎 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache 2.0) — 阿里通义千问团队
>
> 论文: https://arxiv.org/abs/2601.15621 · PyPI: `qwen-tts`

---

## 功能特色

| 类别 | 功能 | 说明 |
|------|------|------|
| 🎤 | **音色克隆** | 上传 3 秒以上参考音频，支持 ICL（需文本）和 x-vector（无需文本） |
| 🔊 | **长文本分段合成** | 自动按句子边界切割，每段独立生成，质量可控 |
| 🎯 | **逐段试听** | 每段可单独生成试听，不满意的段可单独替换 |
| 🔄 | **会话管理** | 自动保存每次生成为时间戳文件夹，可加载历史会话继续编辑 |
| 🎛️ | **生成参数可控** | 温度、Top-K/P、重复惩罚、韵律微调（Subtalker） |
| 🔧 | **音频后处理** | 语速调节、段间停顿、气口插入 |
| 💻 | **CLI 批处理** | 命令行模式，支持并行生成、断点续传 |
| 🧹 | **音频质量优化** | 自动检测并裁切开头不稳定能量段，RMS 音量归一化，软限幅 |
| ✅ | **ASR 自动校验** | 用 Whisper 识别段头段尾缺字，瑕疵段自动重试（适合隔夜任务） |
| 🍎 | **Apple Silicon 优化** | 自动检测 MPS，设置最佳环境变量，每段清理 GPU 缓存防 OOM |
| 🎛️ | **右侧控制台** | 实时显示工作流进度、已用时间、各段生成耗时、统计信息 |
| 📐 | **UI 等高布局** | 左栏每行所有控件（下拉框、输入框、按钮、滑块）自动拉伸到统一高度 |

## 快速开始

### 1. 安装环境

```bash
bash setup.sh
```

### 2. 下载模型

```bash
# 方式一：国内使用 hf-mirror
HF_ENDPOINT=https://hf-mirror.com python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base')
"

# 方式二：能访问 HuggingFace 则自动缓存
```

### 3. 启动 WebUI

```bash
source venv/bin/activate
python webui.py
# 浏览器打开 http://127.0.0.1:7860
```

### 4. 使用流程

```
① 加载模型 → ② 上传参考音频 + 创建音色 →
③ 输入文本 → 预览分段 →
④ 逐段生成试听 → 或「生成所有段落」→
⑤ 调整后处理参数 → 合并输出
```

右侧控制台实时跟踪当前工作流阶段、已用时间、各段生成耗时及统计信息。

---

## WebUI 会话管理

每次生成分段时，WebUI 会自动在 `outputs/` 下创建以生成时间命名的文件夹（如 `outputs/202604261954/`），包含：

- `seg_0000.wav`, `seg_0001.wav`, ... — 每个分段的独立音频
- `session.json` — 元数据（源文本、分段文本、各段状态）
- `merged.wav` — 合并后的最终音频

**历史会话加载**：在 WebUI 顶部的会话下拉框中选择历史记录，即可恢复该次生成的所有分段内容和音频，继续编辑或重新后处理。

## WebUI 右侧控制台

WebUI 右侧面板实时显示工作流状态与统计信息：

- **工作流进度**：6 步引导（加载模型 → 创建音色 → 输入文本 → 预览分段 → 逐段生成 → 后处理合并），自动高亮当前阶段
- **统计信息**：已用时间、总字数、总段数、已生成段数、当前会话名
- **各段用时**：按生成顺序显示每段耗时（最近 20 段），方便监控生成效率

所有交互操作（加载模型、创建音色、预览分段、生成音频、合并输出）都会自动刷新控制台。

---

## CLI 命令行

```bash
source venv/bin/activate

# ---- 基础用法 ----
python long_tts.py \
    --ref-audio 参考音频.wav \
    --ref-text "参考音频的文字内容" \
    --input 小说.txt \
    --output 有声书.wav

# ---- x-vector 模式（无需参考文本） ----
python long_tts.py \
    --ref-audio 参考音频.wav \
    --xvec \
    --input 小说.txt \
    --output 有声书.wav

# ---- ASR 自动校验（隔夜任务推荐） ----
# 自动检测段头段尾缺字，瑕疵段自动重试
# 需要安装 openai-whisper（pip install openai-whisper）
python long_tts.py \
    --ref-audio 参考音频.wav --ref-text "..." \
    --input 小说.txt --output 有声书.wav \
    --verify

# ---- CPU 并行生成（x86 平台） ----
python long_tts.py \
    --ref-audio 参考音频.wav --ref-text "..." \
    --input 小说.txt --output 有声书.wav \
    --device cpu --batch 4

# ---- 常用参数组合 ----
python long_tts.py \
    --ref-audio 参考音频.wav --ref-text "参考音频的文字内容" \
    --input 小说.txt --output 有声书.wav \
    --max-seg-len 350 \
    --temperature 0.9 --top-k 50 \
    --speed 0.95 --segment-gap 1.5 --breathing-pause 0.25 \
    --resume --verify
```

### CLI 参数一览

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ref-audio` | — | 参考音频路径（必填） |
| `--ref-text` | — | 参考音频文字（ICL 模式必填） |
| `--xvec` | — | x-vector 模式（无需参考文本） |
| `--input` / `-i` | — | 输入文本文件（不指定则从 stdin 读取） |
| `--output` / `-o` | `output.wav` | 输出音频路径 |
| `--model` | `Qwen3-TTS-12Hz-1.7B-Base` | 模型 ID 或本地路径 |
| `--device` | 自动检测 | MPS(Apple) / CUDA(NVIDIA/AMD ROCm) / cpu |
| `--dtype` | `float16` | 模型精度：float32 / float16 / bfloat16 |
| `--max-seg-len` | 350 | 每段最大字符数 |
| `--temperature` | 0.9 | 采样温度 |
| `--top-k` | 50 | Top-K 采样 |
| `--top-p` | 1.0 | Top-P 采样 |
| `--speed` | — | 语速倍率（交互式选择） |
| `--segment-gap` | — | 段间停顿秒数 |
| `--breathing-pause` | — | 句间气口停顿秒数 |
| `--batch` / `-b` | 1 | 并行生成数（仅 CPU 生效） |
| `--resume` | — | 断点续传 |
| `--verify` | — | ASR 自动校验 + 瑕疵段重试 |
| `--verify-max-retries` | 3 | 单段最大校验重试次数 |
| `--keep-segments` | — | 保留各分段独立文件 |
| `--quiet` / `-q` | — | 减少输出信息 |
| `--dry-run` | — | 仅显示分段结果，不生成 |
| `--save-ref` | — | 将参考音频保存到历史 |

---

## 跨平台支持

| 平台 | GPU | Device | 状态 | 说明 |
|------|-----|--------|------|------|
| macOS Apple Silicon | M1/M2/M3/M4 | `mps` (自动) | ✅ 最优 | GPU 推理，自动内存优化 |
| Linux + NVIDIA | 任何 NVIDIA GPU | `cuda:0` | ✅ 最优 | 支持 Flash Attention 2 |
| Linux + AMD | RX 9070 系列等 | `cuda:0` | ✅ 可用 | 需 ROCm 6.4.1+，暴露 CUDA 接口 |
| Windows / Linux | 任何（纯 CPU） | `cpu` | ✅ 可用 | 可用 `--batch` 并行加速 |

### AMD ROCm（RX 9070 GRE / 9070 XT 等）

ROCm 6.4.1+ 已原生支持 RDNA 4 架构（gfx1200/gfx1201）。在 Linux 上安装 ROCm 后：

```bash
# PyTorch 使用 CUDA 接口（ROCm 兼容）
python long_tts.py \
    --device cuda:0 --dtype float16 \
    --ref-audio 参考音频.wav --ref-text "..." \
    --input 小说.txt --output 有声书.wav
```

> **注意**：ROCm 在 PyTorch 中暴露标准的 `torch.cuda` 接口，所以 `--device cuda:0` 即可识别 AMD GPU。

### CPU 并行加速

GPU 推理非线程安全（MPS/CUDA 均不支持并行），但 CPU 模式可以：

```bash
# CPU 上 4 路并行，约 2-3x 加速
python long_tts.py --device cpu --batch 4 ...
```

---

## 音频质量优化

### 能量检测 + 动态裁切

针对 Qwen3-TTS 生成的音频开头不稳定问题（爆音、喷麦、声音由小变大），实现基于 VAD 的能量检测：

1. **自适应阈值**：`threshold = 峰值能量 × 0.35`，不依赖绝对值
2. **持续确认**：要求能量持续高于阈值至少 100ms，避免裁切掉正常语音
3. **动态裁切**：物理移除不稳定前导部分，而非仅压低音量
4. **短淡入**：裁切后 15ms 余弦淡入，消除 click

### ASR 自动校验（`--verify`）

利用 Whisper `base` 模型对生成的每段音频做段头段尾校验：

- 转录音频 → 提取前 4 个字和后 4 个字
- 与源文本对比 → 缺字则标记为瑕疵
- 自动重试（最多 3 次）→ 微调 temperature 增加随机性
- 输出校验报告：✅ 一次通过 / ↻ 重试修正 / ⚠️ 仍失败

**性能开销**：base 模型转录约 0.2-0.3x 实时，128 段任务约增加 5-8 分钟开销（~3%），隔夜任务完全可忽略。

---

## Apple Silicon 专项优化

自动检测 ARM64 macOS，在导入 PyTorch 前设置：

```python
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
```

- 禁用 MPS 内存水位限制，避免 CPU-GPU 频繁交换
- 允许不支持的算子回退到 CPU
- 每段生成后调用 `torch.mps.empty_cache()`，防止多段累积 OOM

---

## 项目结构

```
qwen-tts-book/
├── webui.py          # WebUI 界面（Gradio + 会话管理 + 控制台）
├── long_tts.py       # CLI 长文本生成工具（所有核心逻辑）
├── setup.sh          # 环境安装脚本
├── start.sh          # 一键启动脚本
├── README.md         # 本文件
├── ref_history.py    # 参考音频历史记录管理
├── ref_history.json  # 参考音频历史数据
└── outputs/          # WebUI 会话输出（自动生成）
    └── 202604261954/ # 每次生成为时间戳文件夹
        ├── session.json
        ├── seg_0000.wav
        ├── seg_0001.wav
        ├── ...
        └── merged.wav
```

## 模型说明

| 模型 | 参数 | 说明 |
|------|------|------|
| `Qwen3-TTS-12Hz-1.7B-Base` | 1.7B | 默认，最佳音质 |
| `Qwen3-TTS-12Hz-0.6B-Base` | 0.6B | 轻量版，生成更快 |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1.7B | 预设音色 + 指令模式 |

## 许可证

本项目代码基于 Apache 2.0 开源。核心语音合成引擎 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 同样使用 Apache 2.0 许可证。

> 本项目完全由 AI 托管生成与发布，如有问题，敬请谅解。
