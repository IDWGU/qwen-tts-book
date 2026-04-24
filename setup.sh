#!/bin/bash
set -e

echo "=== Qwen3-TTS 有声书制作环境安装脚本 ==="
echo ""

# 使用 Homebrew Python 3.11
PYTHON="/opt/homebrew/bin/python3.11"
if [ ! -f "$PYTHON" ]; then
    echo "未找到 $PYTHON，尝试使用 system python3..."
    PYTHON="python3"
fi

echo "使用 Python: $($PYTHON --version)"

# 创建虚拟环境
VENV_DIR="$(cd "$(dirname "$0")" && pwd)/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "创建虚拟环境..."
    $PYTHON -m venv "$VENV_DIR"
    echo "虚拟环境已创建: $VENV_DIR"
else
    echo "虚拟环境已存在: $VENV_DIR"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"

# 升级 pip
pip install --upgrade pip -q

# 安装依赖
echo ""
echo "安装 PyTorch (Apple Silicon MPS 支持)..."
pip install --pre torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
    -q

echo ""
echo "安装 Qwen3-TTS 及相关依赖..."

# 安装 transformers 等核心依赖
pip install \
    "transformers==4.57.3" \
    "accelerate==1.12.0" \
    "librosa" \
    "soundfile" \
    "einops" \
    "onnxruntime" \
    "sentencepiece" \
    -q

# 安装 qwen-tts 包 (从 PyPI)
echo "从 PyPI 安装 qwen-tts..."
pip install -U qwen-tts -q

echo ""
echo "=== 安装完成 ==="
echo ""
echo "使用方式:"
echo "  1. 激活环境: source venv/bin/activate"
echo "  2. 运行脚本: python long_tts.py [参数]"
echo ""
echo "快速开始:"
echo "  python long_tts.py --ref-audio 参考音频.wav --ref-text \"参考文本\" --input 小说.txt --output 有声书.wav"
echo ""
