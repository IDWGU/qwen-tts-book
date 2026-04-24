# 🎧 Qwen3-TTS 有声书生成器

基于阿里 **Qwen3-TTS**（Apache 2.0）的开源有声书制作工具，支持**音色克隆**与**长文本分段合成**。

## 特性

- **音色克隆** — 上传一段参考音频（3 秒以上），即可克隆其音色
- **长文本合成** — 智能分段，自动处理超长文本，支持断点续传
- **WebUI 操作** — 基于 Gradio 的图形界面，无需写代码
- **CLI 命令行** — 适合批量处理和自动化脚本
- **后处理丰富** — 语速调节、段间停顿、句间气口，让有声书更自然
- **Apple Silicon 优化** — 原生支持 MPS 加速

## 快速开始

### 1. 安装环境

```bash
bash setup.sh
source venv/bin/activate
```

### 2. 使用 WebUI

```bash
python webui.py
# 浏览器打开 http://127.0.0.1:7860
```

### 3. 使用命令行

```bash
python long_tts.py \
    --ref-audio 参考音频.wav \
    --ref-text "参考音频的文字内容" \
    --input 小说.txt \
    --output 有声书.wav
```

## 项目结构

```
qwen-tts-book/
├── setup.sh          # 环境安装脚本
├── long_tts.py       # 核心长文本 TTS 工具（CLI）
├── webui.py          # Gradio Web 图形界面
└── README.md         # 本文件
```

## 依赖

| 依赖 | 用途 |
|------|------|
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (qwen-tts) | 核心 TTS 模型与音色克隆 |
| Gradio | WebUI 界面 |
| PyTorch | 深度学习推理 |
| librosa | 音频分析与处理 |
| soundfile | 音频文件读写 |

## 鸣谢

- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — 阿里通义千问团队开源的 TTS 模型（Apache 2.0）
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — 模型加载框架

## 许可证

本项目代码基于 **Apache 2.0** 许可证开源。

> 本项目完全由 AI 托管生成与发布，如有问题，敬请谅解。
