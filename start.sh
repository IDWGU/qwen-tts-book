#!/bin/bash
set -e

# ============================================================
#  Qwen3-TTS 有声书生成器 · 一键启动脚本
# ============================================================
#  首次运行会自动：安装环境 → 下载模型 → 启动
#  之后每次运行：直接启动（跳过安装）
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
MODEL_ID="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"

# ---- 颜色输出 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
err()   { echo -e "${RED}[ERR]${NC}   $1"; }

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   🎧 Qwen3-TTS 有声书生成器 一键启动     ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

# ============================================================
# Step 1: 检查/创建虚拟环境
# ============================================================
if [ ! -d "$VENV_DIR" ]; then
    info "虚拟环境不存在，开始创建..."

    # 查找 Python
    if [ -f "/opt/homebrew/bin/python3.11" ]; then
        SYSTEM_PYTHON="/opt/homebrew/bin/python3.11"
    else
        SYSTEM_PYTHON="python3"
    fi

    info "使用 Python: $SYSTEM_PYTHON ($($SYSTEM_PYTHON --version 2>&1))"
    "$SYSTEM_PYTHON" -m venv "$VENV_DIR"
    ok "虚拟环境已创建"

    # 升级 pip
    "$PIP" install --upgrade pip -q
    ok "pip 已升级"

    # 安装 PyTorch（Apple Silicon MPS）
    info "安装 PyTorch (Apple Silicon)..."
    "$PIP" install --pre torch torchvision torchaudio \
        --extra-index-url https://download.pytorch.org/whl/nightly/cpu -q 2>&1 | tail -1
    ok "PyTorch 安装完成"

    # 安装核心依赖
    info "安装 Qwen3-TTS 核心依赖..."
    "$PIP" install \
        "transformers==4.57.3" \
        "accelerate==1.12.0" \
        "librosa" \
        "soundfile" \
        "einops" \
        "onnxruntime" \
        "sentencepiece" \
        "gradio" \
        -q 2>&1 | tail -1
    ok "核心依赖安装完成"

    # 安装 qwen-tts
    info "安装 qwen-tts..."
    "$PIP" install -U qwen-tts -q 2>&1 | tail -1
    ok "qwen-tts 安装完成"

    # 安装 pyrubberband（高质量变速）
    info "安装 RubberBand 变速引擎..."
    if command -v brew &>/dev/null; then
        brew install rubberband 2>/dev/null && ok "rubberband 已安装 (brew)" || warn "rubberband brew 安装失败，将使用 fallback 变速"
    else
        warn "未找到 Homebrew，将使用 fallback 变速"
    fi
    "$PIP" install pyrubberband -q 2>&1 | tail -1
    ok "pyrubberband 安装完成"

    ok "===== 环境安装完成 ====="
else
    ok "虚拟环境已存在，跳过安装"
fi

# ============================================================
# Step 2: 检查模型是否已下载
# ============================================================
MODEL_CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"
MODEL_DOWNLOADED=false

if [ -d "$MODEL_CACHE" ] && [ -f "$MODEL_CACHE/refs/main" ]; then
    SNAPSHOT_HASH=$(cat "$MODEL_CACHE/refs/main" 2>/dev/null)
    if [ -n "$SNAPSHOT_HASH" ] && [ -d "$MODEL_CACHE/snapshots/$SNAPSHOT_HASH" ]; then
        MODEL_DOWNLOADED=true
    fi
fi

if [ "$MODEL_DOWNLOADED" = false ]; then
    echo ""
    warn "模型尚未下载: $MODEL_ID"
    info "正在下载模型（约 3.5GB，首次下载需要几分钟）..."
    echo ""
    echo -e "  ${YELLOW}国内网络建议使用镜像:${NC}"
    echo -e "  ${YELLOW}  如果下载慢，请按 Ctrl+C 取消，然后运行:${NC}"
    echo -e "  ${YELLOW}  HF_ENDPOINT=https://hf-mirror.com bash start.sh${NC}"
    echo ""

    # 检查是否设置了 HF_ENDPOINT 镜像
    if [ -n "$HF_ENDPOINT" ]; then
        info "使用镜像: $HF_ENDPOINT"
    fi

    "$PYTHON" -c "
from huggingface_hub import snapshot_download
import sys
print(f'下载 {sys.argv[1]}...')
snapshot_download(sys.argv[1])
print('下载完成！')
" "$MODEL_ID"
    ok "模型下载完成"
else
    ok "模型已缓存，跳过下载"
fi

# ============================================================
# Step 3: 选择启动模式
# ============================================================
echo ""
echo -e "${CYAN}请选择启动模式:${NC}"
echo "  [1] 🌐 WebUI 界面（Gradio 浏览器操作）"
echo "  [2] 💻 CLI  命令行（终端直接合成）"
echo "  [3] 🚪 退出"
echo ""

read -p "请输入选项 [1-3] (默认 1): " MODE
MODE=${MODE:-1}

case "$MODE" in
    1)
        echo ""
        info "正在启动 WebUI..."
        echo -e "  ${GREEN}浏览器将自动打开: http://127.0.0.1:7860${NC}"
        echo ""
        "$PYTHON" "$SCRIPT_DIR/webui.py" "$@"
        ;;
    2)
        echo ""
        info "CLI 命令行模式"
        echo ""
        echo -e "${CYAN}用法示例:${NC}"
        echo "  python long_tts.py --ref-audio 参考音频.wav --ref-text \"参考文本\" --input 小说.txt --output 有声书.wav"
        echo ""
        echo -e "${CYAN}常用参数:${NC}"
        echo "  --model       模型名称 (默认: Qwen/Qwen3-TTS-12Hz-1.7B-Base)"
        echo "  --ref-audio   参考音频路径（必填）"
        echo "  --ref-text    参考音频文字（ICL 模式必填）"
        echo "  --xvec        x-vector only 模式（无需参考文本）"
        echo "  --input       输入文本文件"
        echo "  --output      输出音频文件 (默认: output.wav)"
        echo "  --max-seg-len 每段最大字符数 (默认: 350)"
        echo "  --speed       语速 (默认: 0.9)"
        echo "  --language    语言 (默认: Auto)"
        echo "  --resume      断点续传"
        echo "  --dry-run     仅预览分段"
        echo "  --quiet       减少输出"
        echo ""
        echo -e "${CYAN}快速开始:${NC}"
        echo "  $PYTHON $SCRIPT_DIR/long_tts.py --ref-audio <音频.wav> --ref-text \"<参考文本>\" --input <小说.txt> --output <输出.wav>"
        echo ""

        # 如果有入参则直接执行，否则进入交互式 shell
        if [ $# -gt 0 ]; then
            "$PYTHON" "$SCRIPT_DIR/long_tts.py" "$@"
        else
            # 进入虚拟环境的 bash，方便用户手动执行命令
            echo -e "${CYAN}已进入虚拟环境，可以直接运行 python long_tts.py ...${NC}"
            echo -e "${CYAN}输入 exit 退出${NC}"
            echo ""
            exec "$VENV_DIR/bin/bash" --rcfile <(echo "PS1='(qwen-tts) \[\e[36m\]\w\[\e[0m\] \$ '")
        fi
        ;;
    3)
        echo "👋 再见!"
        exit 0
        ;;
    *)
        err "无效选项，请选择 1-3"
        exit 1
        ;;
esac
