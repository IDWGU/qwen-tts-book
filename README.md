# 🎧 Qwen3-TTS 有声书生成器

基于阿里 Qwen3-TTS 的开源语音合成工具，支持音色克隆、长文本分段合成、逐段试听与合并。

> 基于 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache 2.0) — 阿里通义千问团队
>
> 论文: https://arxiv.org/abs/2601.15621 · PyPI: `qwen-tts`

---

## 功能特色

- **🎤 音色克隆** — 上传 3 秒以上参考音频即可克隆音色，支持 ICL（需文本）和 x-vector（无需文本）两种模式
- **🔊 长文本分段合成** — 自动按句子边界切割文本，每段独立生成，质量可控
- **🎯 逐段试听与合并** — 每段可单独生成试听，不满意重新生成该段，确认后合并
- **🎛️ 生成参数可控** — 温度、Top-K/P、重复惩罚、韵律微调（Subtalker）
- **🔧 音频后处理** — 语速调节、段间停顿、气口插入
- **🔍 声音特征分析** — 自动分析参考音频的音高/音量/语速，推荐风格描述
- **🌐 WebUI 界面** — 基于 Gradio，直观易用
- **💻 CLI 工具** — 命令行模式，支持批量处理和断点续传

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
④ 逐段生成试听 → ⑤ 合并输出
```

## CLI 命令行

```bash
source venv/bin/activate

# 基础用法
python long_tts.py \
    --ref-audio 参考音频.wav \
    --ref-text "参考音频的文字内容" \
    --input 小说.txt \
    --output 有声书.wav

# x-vector 模式（无需参考文本）
python long_tts.py \
    --ref-audio 参考音频.wav \
    --xvec \
    --input 小说.txt \
    --output 有声书.wav
```

## 项目结构

```
qwen-tts-book/
├── webui.py       # WebUI 界面（Gradio）
├── long_tts.py    # CLI 长文本生成工具
├── setup.sh       # 环境安装脚本
├── README.md      # 本文件
└── venv/          # Python 虚拟环境
```

## 模型说明

默认使用 `Qwen3-TTS-12Hz-1.7B-Base`（1.7B 参数，最佳音质），也支持：

- `Qwen/Qwen3-TTS-12Hz-0.6B-Base` — 轻量版
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` — 预设音色 + 指令模式

## 许可证

本项目代码基于 Apache 2.0 开源。核心语音合成引擎 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 同样使用 Apache 2.0 许可证。

> 本项目完全由 AI 托管生成与发布，如有问题，敬请谅解。
