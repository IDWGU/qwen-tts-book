#!/usr/bin/env python3
"""
Qwen3-TTS 长文本有声书生成工具
==================================

功能:
  1. 加载 Qwen3-TTS Base 模型 + 音色克隆
  2. 读取长文本（文件或标准输入）
  3. 智能分段（按句子/段落边界切割）
  4. 逐段生成语音（保持音色一致）
  5. 合并所有音频段为单一文件
  6. 支持断点续传

用法:
  # 基础用法：音色克隆 + 长文本合成
  python long_tts.py \\
      --ref-audio 参考音频.wav \\
      --ref-text "参考音频的文字内容" \\
      --input 小说.txt \\
      --output 有声书.wav

  # x-vector 模式（只需参考音频，不需文字）
  python long_tts.py \\
      --ref-audio 参考音频.wav \\
      --xvec \\
      --input 小说.txt \\
      --output 有声书.wav

  # 从标准输入读取文本
  cat 小说.txt | python long_tts.py --ref-audio 参考音频.wav --ref-text "..."

  # 指定分段长度和语言
  python long_tts.py \\
      --ref-audio 参考音频.wav --ref-text "..." \\
      --input 小说.txt --output 有声书.wav \\
      --max-seg-len 300 --language Chinese

  # 指定模型
  python long_tts.py ... --model Qwen/Qwen3-TTS-12Hz-1.7B-Base

  # 断点续传（自动检测已有分段）
  python long_tts.py ... --resume

  # 仅查看文本分段结果
  python long_tts.py --input 小说.txt --dry-run
"""

import argparse
import gc
import os
import re
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import concurrent.futures

# =========================================================================
# Apple Silicon (MPS) 自动检测与优化 — 必须在任何 torch import 之前设置
# =========================================================================
_IS_APPLE_SILICON = os.uname().machine == "arm64"
if _IS_APPLE_SILICON:
    # PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0: 禁用 MPS 内存水位限制，允许模型
    # 完全加载到 GPU，避免频繁 CPU-GPU 内存交换导致的性能下降。
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    # PYTORCH_ENABLE_MPS_FALLBACK=1: 允许不支持 MPS 的操作自动回退到 CPU，
    # 避免因算子不兼容导致的崩溃，确保模型能正常加载。
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import ref_history

import librosa
import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel

# 提前导入 pyrubberband（而非在 adjust_speed 函数内重复 import）
try:
    import pyrubberband as pyrb
    HAS_PYRUBBERBAND = True
except ImportError:
    HAS_PYRUBBERBAND = False
    if not os.environ.get("QWTTS_QUIET_NO_PYRB"):
        import sys as _sys
        _sys.stderr.write(
            "⚠️  pyrubberband 未安装，使用内置 OLA 算法变速。\n"
            "   安装 pyrubberband 可获得更佳音质：pip install pyrubberband\n"
            "   设置环境变量 QWTTS_QUIET_NO_PYRB=1 可屏蔽此提示。\n"
        )

# =========================================================================
# 1. 文本分段
# =========================================================================

# 次级分隔符（用于超长句的强制拆分）
SUB_SEPARATOR = re.compile(r"([，、,、\s]{1,2})")


# 句子边界模式：一次扫描完成分割，避免逐字符 join
_SENTENCE_BOUNDARY = re.compile(r'[^。！？.!?\n]*[。！？.!?\n]')


def split_sentences(text: str) -> List[str]:
    """将文本按句子边界切分，保留标点符号在句尾。

    使用 regex 一次扫描，在句子结束标点处切分：
      - 中文：。！？
      - 英文：.!?
      - 换行：\\n

    注意：用原始（未 strip）字符串长度追踪 matched_len，
    避免 strip 偏移导致 text[matched_len:] 索引错误。

    时间复杂度 O(n)，内存占用 O(n)。
    """
    raw_sentences = _SENTENCE_BOUNDARY.findall(text)
    # 去掉空串和纯空白，同时记录原始长度
    sentences: List[str] = []
    matched_len = 0
    for s in raw_sentences:
        stripped = s.strip()
        if stripped:
            sentences.append(stripped)
            matched_len += len(s)  # 原始长度（含前导/尾随空格）
    remaining = text[matched_len:].strip()
    if remaining:
        if sentences and len(remaining) < 10:
            sentences[-1] += remaining
        else:
            sentences.append(remaining)
    return sentences


def segment_text(
    text: str,
    max_chars: int = 500,
) -> List[str]:
    """将长文本分割成适合 TTS 生成的段落。

    策略:
      1. 先按句子分割
      2. 将短句子合并，直到接近 max_chars
      3. 单个超长句子强制截断（按逗号/空格等次级分隔符）

    时间复杂度 O(n)，使用 list+join 避免 Python 字符串 O(n²) 拼接。

    Args:
        text: 输入文本
        max_chars: 每段最大字符数

    Returns:
        分段后的文本列表
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    segments: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # 单句超长：先用次级分隔符（逗号/空格）拆分，如果拆不动则按字符数强制切割
        if len(sent) > max_chars:
            # 先把当前累积的段落提交
            if current_parts:
                segments.append("".join(current_parts).strip())
                current_parts = []
                current_len = 0
            # 尝试用逗号/空格拆分
            sub_sentences = SUB_SEPARATOR.split(sent)
            # 如果 regex 拆不动（整个句子是连续的），按字符数强制切分
            if len(sub_sentences) <= 1:
                for i in range(0, len(sent), max_chars):
                    segments.append(sent[i:i + max_chars])
                continue
            sub_parts: List[str] = []
            sub_len = 0
            for part in sub_sentences:
                if sub_len + len(part) > max_chars and sub_parts:
                    segments.append("".join(sub_parts).strip())
                    sub_parts = [part]
                    sub_len = len(part)
                else:
                    sub_parts.append(part)
                    sub_len += len(part)
            if sub_parts:
                segments.append("".join(sub_parts).strip())
            continue

        # 正常情况：累积到 max_chars 就切分（list+join 避免 O(n²)）
        sent_len = len(sent)
        if current_len + sent_len > max_chars and current_parts:
            segments.append("".join(current_parts).strip())
            current_parts = [sent]
            current_len = sent_len
        else:
            current_parts.append(sent)
            current_len += sent_len

    # 最后一段
    if current_parts:
        segments.append("".join(current_parts).strip())

    return segments


# =========================================================================
# 2. 音频处理（变速 + 停顿 + 合并）
# =========================================================================


def _ola_time_stretch(wav: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """OLA (Overlap-Add) 变速算法，专门针对语音优化。

    相比 librosa 相位声码器（phase vocoder），OLA 不修改信号的相位信息，
    通过重叠窗口的增删实现时间缩放，完全避免金属感的"电音"。

    原理：
      - 将音频切成 30ms 的窗口，75% 重叠的汉宁窗
      - 变速时调整合成步长（hs = ha / speed）
      - 累加后归一化，消除因重叠产生的幅度变化

    Args:
        wav: 输入音频 (float32, [-1, 1])
        sr: 采样率
        speed: 速度倍率 (>1 加快, <1 减慢)

    Returns:
        变速后的音频
    """
    if abs(speed - 1.0) < 0.01:
        return wav

    n = len(wav)
    ws = int(0.03 * sr + 0.5)   # 30ms 窗口（四舍五入）
    if ws < 64:
        ws = min(256, n)          # 极短音频保护
    ha = max(ws // 4, 1)          # 分析步长（75% 重叠）
    hs = max(int(ha / speed + 0.5), 1)  # 合成步长

    window = np.hanning(ws).astype(np.float64)

    # 输出长度 ≈ n / speed，留一个窗口余量
    out_len = int(np.ceil(n / speed)) + ws
    out = np.zeros(out_len, dtype=np.float64)
    norm = np.zeros(out_len, dtype=np.float64)

    ipos = 0
    opos = 0
    while ipos + ws <= n:
        seg = wav[ipos:ipos + ws].astype(np.float64) * window
        end = opos + ws
        if end > out_len:
            break
        out[opos:end] += seg
        norm[opos:end] += window
        ipos += ha
        opos += hs

    # 归一化（除以重叠累积的窗函数值）
    norm[norm < 1e-10] = 1.0
    result = (out / norm).astype(np.float32)

    # 削波保护（OLA 累加可能导致峰值略超 1.0）
    peak = float(np.max(np.abs(result)))
    if peak > 1.0:
        result /= peak

    # 裁剪到有效范围
    valid = min(opos + ws, out_len)
    return result[:valid]


def adjust_speed(wav: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """调整语速，不变调。

    优先使用 RubberBand 算法（pyrubberband），如果不支持则使用
    自实现的 OLA 算法。两者均比 librosa 相位声码器音质好，无电音。

    Args:
        wav: 音频数据
        sr: 采样率
        speed: 速度倍率 (0.5=慢一倍, 1.0=原速, 1.5=快50%)

    Returns:
        变速后的音频
    """
    if abs(speed - 1.0) < 0.01:
        return wav
    if HAS_PYRUBBERBAND:
        # RubberBand 算法 — 行业最优质的不变调变速，无电音
        return pyrb.time_stretch(wav.astype(np.float64), sr, speed)
    # OLA fallback — 无相位声码器电音问题
    return _ola_time_stretch(wav, sr, speed)


def _detect_onset(
    wav: np.ndarray,
    sr: int,
    search_window_ms: float = 2000,
    energy_threshold_ratio: float = 0.35,
    lookahead_ms: float = 10,
    min_trim_ms: float = 30,
    sustain_ms: float = 100,
) -> int:
    """检测音频开头语音稳定起始点，裁切掉「由小变大」的渐起段。

    解决 Qwen3-TTS 开头前几个字声音由小变大的问题。
    核心改进：
      - 自适应阈值：基于搜索窗口内的峰值能量（而非固定值）
      - 持续确认：要求能量持续高于阈值至少 sustain_ms
        → 确保切到「音量稳定」的位置，而非 ramp-up 开端

    策略：
      1. 取前 search_window_ms 的波形
      2. 计算短时能量包络（abs + 5ms 移动平均平滑）
      3. 找到窗口内的峰值能量
      4. 自适应阈值 = 峰值 × energy_threshold_ratio
      5. 找到首次超过阈值且持续 sustain_ms 的位置
      6. 在该位置前保留 lookahead_ms 作为过渡
      7. 至少裁掉 min_trim_ms

    Args:
        wav: 输入音频 (float32)
        sr: 采样率 (Hz)
        search_window_ms: 搜索窗口时长（毫秒），默认 2000ms 以覆盖完整 ramp-up
        energy_threshold_ratio: 阈值比例（相对于峰值能量），默认 0.35
        lookahead_ms: onset 前保留的过渡时长（毫秒）
        min_trim_ms: 最小裁切量（毫秒）
        sustain_ms: 持续确认时长（毫秒），默认 100ms

    Returns:
        裁切起始样本索引，即 out = wav[return_value:]
    """
    search_samples = int(search_window_ms * sr / 1000)
    lookahead_samples = int(lookahead_ms * sr / 1000)
    min_trim = int(min_trim_ms * sr / 1000)
    sustain_samples = int(sustain_ms * sr / 1000)

    window = wav[:search_samples]
    if len(window) == 0:
        return 0

    # 短时能量包络：abs → 5ms 移动平均平滑
    envelope = np.abs(window)
    kernel_size = max(1, int(5 * sr / 1000))
    kernel = np.ones(kernel_size) / kernel_size
    envelope = np.convolve(envelope, kernel, mode='same')

    peak_energy = float(np.max(envelope))
    # 音频几乎静音 → 返回最小裁切
    if peak_energy < 1e-8:
        return min(min_trim, len(wav) // 4)

    # 自适应阈值
    threshold = peak_energy * energy_threshold_ratio

    # 找到首次超过阈值且持续 sustain_samples 的位置
    above = (envelope > threshold).astype(np.int8)

    if sustain_samples > 1 and len(above) >= sustain_samples:
        # 卷积法：找到连续 sustain_samples 个样本都超过阈值的位置
        sus_kernel = np.ones(sustain_samples, dtype=np.int8)
        counts = np.convolve(above, sus_kernel, mode='valid')
        valid = np.where(counts >= sustain_samples)[0]
        if len(valid) > 0:
            onset = int(valid[0])
        else:
            # 没有持续段，退化为简单阈值
            first_above = np.where(above == 1)[0]
            if len(first_above) > 0:
                onset = int(first_above[0])
            else:
                return min(min_trim, len(wav) // 4)
    else:
        # sustain_samples <= 1 → 直接用简单阈值
        first_above = np.where(above == 1)[0]
        if len(first_above) > 0:
            onset = int(first_above[0])
        else:
            return min(min_trim, len(wav) // 4)

    trim_point = max(0, onset - lookahead_samples)
    trim_point = max(trim_point, min_trim)
    return trim_point


def level_loudness(
    wav: np.ndarray,
    sr: int,
    target_rms: float = 0.08,
    strength: float = 1.0,
) -> np.ndarray:
    """响度均衡：双时间尺度 + 气口抑制。

    用长程 RMS 参考（1.5s）计算增益，气口等短时低能量事件被平滑滤除，
    不会触发增益放大。再通过短程/长程 RMS 比值定位气口帧，主动压低。

    Args:
        wav: 输入音频 (float32, [-1, 1])
        sr: 采样率
        target_rms: 目标 RMS 电平 (默认 0.08，约 -22dB)
        strength: 均衡力度 0.0~1.0 (默认 1.0)
            0.0 = 无效果
            0.5 = 半程修正，保留部分原始动态
            1.0 = 完全均衡到 target_rms

    Returns:
        均衡后的音频
    """
    out = wav.copy().astype(np.float64)
    n = len(out)
    if n < sr * 0.3:
        return wav

    # ── 1. 短时 RMS 包络（50ms 窗口, 25ms 步进） ──
    frame_len = max(int(0.05 * sr), 1)
    hop = max(int(0.025 * sr), 1)

    frames = []
    positions = []
    for start in range(0, n - frame_len + 1, hop):
        r = float(np.sqrt(np.mean(out[start:start + frame_len] ** 2)))
        frames.append(max(r, 1e-10))
        positions.append(start + frame_len // 2)
    positions.append(n - 1)
    frames.append(frames[-1] if frames else 1e-10)

    rms_env = np.array(frames, dtype=np.float64)
    pos = np.array(positions)

    # ── 2. 平滑 RMS（300ms 窗口，去微波动） ──
    smooth_n = max(int(0.3 * sr / hop), 2)
    kernel = np.ones(smooth_n) / smooth_n
    rms_env = np.convolve(rms_env, kernel, mode="same")

    # ── 3. 长程参考 RMS（1.5s 窗口） ──
    #     气口/呼吸声是短时事件（0.1~0.3s），几乎不影响长程均值
    #     只有持续 1.5s+ 的音量变化才会被修正
    long_n = max(int(1.5 * sr / hop), 2)
    long_kernel = np.ones(long_n) / long_n
    long_rms = np.convolve(rms_env, long_kernel, mode="same")

    # ── 4. 增益曲线（基于长程参考，气口不触发） ──
    gain_raw = (target_rms / long_rms) ** strength
    gain_raw = np.clip(gain_raw, 0.05, 10.0)

    # ── 5. 气口定位 & 主动抑制 ──
    #     短程/长程比值：气口帧比值低（<<0.5），语音帧比值高（~1.0）
    #     ratio = rms_env / long_rms
    #     增加一个小 epsilon 防止 long_rms ≈ 0 时爆炸
    ratio = rms_env / np.maximum(long_rms, 1e-10)
    # 抑制因子：ratio < 0.3 → 全压制 (factor→0)
    #           ratio > 0.6 → 无压制 (factor→1)
    #           中间线性过渡
    suppress = np.clip((ratio - 0.3) / 0.3, 0.0, 1.0)
    # 气口帧 gain 被压低到 20%（比之前 0.3 更低）
    breath_floor = 0.2
    gain_raw = breath_floor + (gain_raw - breath_floor) * suppress

    # ── 6. 平滑增益曲线（避免帧间突变） ──
    smooth_gain_n = max(int(0.1 * sr / hop), 2)
    kernel2 = np.ones(smooth_gain_n) / smooth_gain_n
    gain_raw = np.convolve(gain_raw, kernel2, mode="same")

    # ── 7. 插值到完整音频长度 ──
    gain_full = np.interp(np.arange(n), pos, gain_raw)

    # ── 8. 应用增益 ──
    out *= gain_full

    # ── 9. 软限幅（防止增益放大后的峰值过冲） ──
    threshold = 0.92
    over = np.abs(out) > threshold
    if over.any():
        ratio_lim = 0.3
        sign = np.sign(out)
        excess = np.abs(out) - threshold
        out[over] = sign[over] * (threshold + excess[over] * ratio_lim)

    return out.astype(np.float32)


def normalize_audio(
    wav: np.ndarray,
    sr: int,
    target_peak: float = 0.90,
    fade_in_ms: float = 15,
    max_peak: float = 0.95,
    trim_unstable_onset: bool = True,
) -> np.ndarray:
    """音频归一化：动态裁切不稳定开头 + 短淡入 + 峰值归一化 + 软限幅。

    解决 Qwen3-TTS 生成音频的常见问题：
      - 开头爆音/喷麦 → 能量检测动态裁切 + 15ms 余弦淡入
      - 各段音量不一致 → 峰值归一化到统一电平（比 RMS 更直观，不受汉字间静音影响）
      - 峰值过大 → 软限幅削波
      - 前几个 acoustic frame 不稳定 → 基于 VAD 的 onset 检测自动去除前导噪声

    Args:
        wav: 输入音频 (float32, [-1, 1])
        sr: 采样率
        target_peak: 目标峰值电平 (默认 0.90，约 -0.9dBFS)
        fade_in_ms: 淡入时长毫秒 (默认 15ms，配合裁切使用)
        max_peak: 软限幅阈值 (默认 0.95)
        trim_unstable_onset: 是否开启动态 onset 裁切 (默认 True)

    Returns:
        处理后的音频
    """
    out = wav.copy()

    # 0. 能量检测 → 动态裁切开头不稳定前导部分
    if trim_unstable_onset:
        trim_idx = _detect_onset(out, sr)
        if trim_idx > 0:
            out = out[trim_idx:]

    # 1. 短余弦淡入（消除裁切点可能产生的 click）
    fade_len = int(fade_in_ms * sr / 1000)
    if fade_len > 0 and fade_len < len(out):
        fade_curve = 0.5 * (1 - np.cos(np.pi * np.arange(fade_len) / fade_len))
        out[:fade_len] *= fade_curve

    # 2. 峰值归一化（以最大峰值为基准，不受汉字间间隔影响）
    current_peak = float(np.max(np.abs(out)))
    if current_peak > 1e-10:
        gain = target_peak / current_peak
        out *= min(gain, 3.0)  # 最大增益 3x，防止过度放大噪声

    # 3. 软限幅（防止峰值再次过冲）
    threshold = max_peak
    ratio = 0.5  # 压缩比
    over = np.abs(out) > threshold
    if over.any():
        sign = np.sign(out)
        excess = np.abs(out) - threshold
        out[over] = sign[over] * (threshold + excess[over] * ratio)

    return out.astype(np.float32)


# =============================================================================
# ASR 校验模块（可选依赖：faster-whisper / whisper）
# =============================================================================
# 用于自动检测生成音频的段头/段尾是否缺字，瑕疵段自动重新生成。
# 适合隔夜批量任务场景。

# 本机 Whisper 模型缓存路径（如果存在则不用下载）
_LOCAL_WHISPER_PT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "whisper_models", "base.pt",
)

_ASR_MODEL = None
_ASR_BACKEND = ""


def init_asr(quiet: bool = False) -> bool:
    """延迟初始化 ASR 模型（faster-whisper → whisper 后备）。

    优先使用项目内的本地模型（whisper_models/base.pt），
    避免重复下载。

    Returns:
        True 表示 ASR 可用
    """
    global _ASR_MODEL, _ASR_BACKEND

    # 尝试 faster-whisper（更快，内存更省）
    try:
        from faster_whisper import WhisperModel
        _ASR_MODEL = WhisperModel(
            "base", device="auto", compute_type="int8",
            cpu_threads=4, num_workers=1,
        )
        _ASR_BACKEND = "faster-whisper"
        if not quiet:
            print("   🎤 ASR 校验: faster-whisper base（命中缓存，无需下载）")
        return True
    except ImportError:
        pass

    # 后备：openai-whisper（优先使用本地模型文件）
    try:
        import whisper  # type: ignore
        model_path = _LOCAL_WHISPER_PT if os.path.isfile(_LOCAL_WHISPER_PT) else "base"
        _ASR_MODEL = whisper.load_model(model_path)
        _ASR_BACKEND = "whisper"
        source = "whisper_models/" if os.path.isfile(_LOCAL_WHISPER_PT) else "cache"
        if not quiet:
            print(f"   🎤 ASR 校验: whisper base（{source}）")
        return True
    except ImportError:
        pass

    if not quiet:
        print("   ⚠️  ASR 校验不可用（pip install faster-whisper 或 pip install openai-whisper）")
    return False


def _transcribe(audio_path: str, language: str) -> Tuple[str, float]:
    """对音频进行 ASR 转写。

    Returns:
        (转录文本, 置信度)
    """
    if _ASR_BACKEND == "faster-whisper":
        segs, info = _ASR_MODEL.transcribe(
            audio_path, language=language,
            beam_size=1, vad_filter=True,
        )
        text = "".join(s.text for s in segs).strip()
        confidence = getattr(info, "average_confidence", 0.0)
    elif _ASR_BACKEND == "whisper":
        result = _ASR_MODEL.transcribe(audio_path, language=language)
        text = result["text"].strip()
        segs = result.get("segments", [])
        confidence = sum(s.get("confidence", 0.0) for s in segs) / max(len(segs), 1) if segs else 0.0
    else:
        text, confidence = "", 0.0
    return text, confidence


def _strip_punct(s: str) -> str:
    """去除标点和空格，仅保留中文和字母数字。"""
    return "".join(ch for ch in s if ch.isalnum() or '\u4e00' <= ch <= '\u9fff')


def verify_segment(
    audio_path: str,
    source_text: str,
    language: str = "zh",
    check_chars: int = 4,
) -> Tuple[bool, str, float, str]:
    """ASR 校验单段音频是否完整生成。

    检查项:
      - 段头校验：转录开头是否包含源文本的前 check_chars 个字
      - 段尾校验：转录结尾是否包含源文本的后 check_chars 个字

    注意：
      - 使用子串搜索而非严格等值（应对 ASR 的识别误差）
      - 源文本不足 6 个字时不校验（太短无法判断）

    Args:
        audio_path: 音频文件路径
        source_text: 预期的源文本
        language: 语言代码 (默认 "zh")
        check_chars: 头部/尾部校验的字符数 (默认 4)

    Returns:
        (通过, 转录文本, 置信度, 失败原因描述)
    """
    if _ASR_MODEL is None:
        return True, "", 0.0, "ASR 未初始化，跳过校验"

    # 转写
    transcription, confidence = _transcribe(audio_path, language)

    text_clean = _strip_punct(source_text)
    trans_clean = _strip_punct(transcription)

    # 太短跳过校验（无法判断）
    if len(text_clean) < 6:
        return True, transcription, confidence, "源文本过短，跳过校验"

    reasons = []

    # ---- 段头校验 ----
    head_target = text_clean[:check_chars]
    # 在转录的前 check_chars*3 范围内搜索
    head_search = trans_clean[:check_chars * 3]
    if head_target not in head_search:
        reasons.append(f"段头缺字: 预期「{head_target}」未出现在开头")

    # ---- 段尾校验 ----
    tail_target = text_clean[-check_chars:]
    tail_search = trans_clean[-check_chars * 3:]
    if tail_target not in tail_search:
        reasons.append(f"段尾缺字: 预期「{tail_target}」未出现在末尾")

    passed = len(reasons) == 0
    fail_reason = "；".join(reasons)
    return passed, transcription, confidence, fail_reason


def _transcribe_head_tail(
    audio_path: str,
    language: str,
    head_duration: float = 2.0,
    tail_duration: float = 2.0,
) -> Tuple[str, float]:
    """仅转录音频的头尾部分，大幅减少 Whisper 计算量。

    方案：截取前 head_duration 秒和后 tail_duration 秒，
    用 0.3s 静音拼接后一次性转录。Whisper 会将其识别为两段独立语音。
    对于 2+2=4s 的音频，转录时间为全段的 1/5 ~ 1/10。

    Args:
        audio_path: 音频文件路径
        language: 语言代码
        head_duration: 头部截取秒数
        tail_duration: 尾部截取秒数

    Returns:
        (转录文本, 置信度)
    """
    # 读取音频
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    total_samples = len(audio)

    # 防止段太短时截取重叠
    max_half = min(total_samples // 2, int(head_duration * sr))
    head = audio[:max_half]

    max_tail = min(total_samples // 2, int(tail_duration * sr))
    tail = audio[-max_tail:] if max_tail > 0 else np.array([], dtype=audio.dtype)

    # 拼接头 + 0.3s 静音 + 尾
    silence = np.zeros(int(0.3 * sr), dtype=audio.dtype)
    combined = np.concatenate([head, silence, tail]) if len(tail) > 0 else head

    # 写入临时文件
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    sf.write(tmp_path, combined, sr)

    try:
        text, confidence = _transcribe(tmp_path, language)
    finally:
        os.unlink(tmp_path)

    return text, confidence


def verify_segment_head_tail(
    audio_path: str,
    source_text: str,
    language: str = "zh",
    check_chars: int = 4,
    head_duration: float = 2.0,
    tail_duration: float = 2.0,
) -> Tuple[bool, str, float, str]:
    """ASR 校验，仅转录音频头尾部分。

    与 verify_segment 功能完全一致，但使用 head-tail 转录优化。
    对于隔夜批量任务，此函数应该替代 verify_segment（性能好 5-10 倍）。

    Args:
        同 verify_segment。额外:
        head_duration: 头部截取秒数 (默认 2.0)
        tail_duration: 尾部截取秒数 (默认 2.0)

    Returns:
        (通过, 转录文本, 置信度, 失败原因描述)
    """
    if _ASR_MODEL is None:
        return True, "", 0.0, "ASR 未初始化，跳过校验"

    # 仅转写头尾
    transcription, confidence = _transcribe_head_tail(
        audio_path, language, head_duration, tail_duration,
    )

    text_clean = _strip_punct(source_text)
    trans_clean = _strip_punct(transcription)

    # 太短跳过校验
    if len(text_clean) < 6:
        return True, transcription, confidence, "源文本过短，跳过校验"

    reasons = []

    # ---- 段头校验：在转录的前半段搜索 ----
    head_target = text_clean[:check_chars]
    mid = max(len(trans_clean) // 2, 1)
    if head_target not in trans_clean[:mid]:
        reasons.append(f"段头缺字: 预期「{head_target}」未出现在开头")

    # ---- 段尾校验：在转录的后半段搜索 ----
    tail_target = text_clean[-check_chars:]
    if tail_target not in trans_clean[mid:]:
        reasons.append(f"段尾缺字: 预期「{tail_target}」未出现在末尾")

    passed = len(reasons) == 0
    fail_reason = "；".join(reasons)
    return passed, transcription, confidence, fail_reason


def make_silence(duration_s: float, sr: int) -> np.ndarray:
    """生成一段静音。"""
    n = int(duration_s * sr)
    return np.zeros(n, dtype=np.float32)


def merge_audio_segments(
    segments: List[Tuple[np.ndarray, int]],
    segment_gap: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """将多个音频段合并为一个，段间可插入静音间隔。

    预分配输出数组，避免 repeated np.concatenate（O(n) 分配 vs O(n²) 复制）。

    Args:
        segments: [(wav_array, sample_rate), ...]
        segment_gap: 段间静音间隔（秒）

    Returns:
        (merged_wav, sample_rate)
    """
    if not segments:
        raise ValueError("没有音频段可合并")

    sr = segments[0][1]
    gap_samples = int(segment_gap * sr)

    # 预计算总长度，一次分配
    total = sum(len(wav) for wav, _ in segments)
    total += gap_samples * (len(segments) - 1)

    out = np.empty(total, dtype=np.float32)
    pos = 0
    for i, (wav, _) in enumerate(segments):
        if i > 0 and gap_samples > 0:
            # gap 区域已在 np.empty 中，显式清零（或直接用 np.zeros 避免）
            out[pos:pos + gap_samples] = 0.0
            pos += gap_samples
        n = len(wav)
        out[pos:pos + n] = wav
        pos += n
    return out, sr

def insert_breathing_pauses(
    wav: np.ndarray,
    text: str,
    sr: int,
    pause_s: float = 0.3,
) -> np.ndarray:
    """在句子边界插入短停顿，模拟气口。

    根据文本中的句子结束标点，在音频对应位置插入停顿。
    采用按比例估算位置的方式（非精准对齐）。

    预分配输出数组避免 repeated np.concatenate：
      旧方案：每句话创建 2 个临时 ndarray（seg + pause），最后 concat → O(n²) 复制
      新方案：1 次 np.empty 分配 + 内存视图切片填充 → O(n)

    Args:
        wav: 音频数据
        text: 对应的文本
        sr: 采样率
        pause_s: 停顿秒数

    Returns:
        带气口的音频
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return wav

    total_chars = len(text)
    if total_chars == 0:
        return wav

    pause_samples = int(pause_s * sr)

    # 第 1 遍：计算每个句子在音频中的起止采样点 + 输出总长度
    boundaries: List[Tuple[int, int]] = []
    current_char = 0
    wav_len = len(wav)
    for sent in sentences:
        start_ratio = current_char / total_chars
        end_ratio = (current_char + len(sent)) / total_chars
        start = int(start_ratio * wav_len)
        end = int(end_ratio * wav_len)
        boundaries.append((start, end))
        current_char += len(sent)

    # 预分配输出数组
    seg_total = sum(end - start for start, end in boundaries)
    out = np.empty(seg_total + pause_samples * (len(boundaries) - 1), dtype=np.float32)

    # 第 2 遍：填充
    pos = 0
    for idx, (start, end) in enumerate(boundaries):
        seg_len = end - start
        out[pos:pos + seg_len] = wav[start:end]
        pos += seg_len
        if idx < len(boundaries) - 1 and pause_samples > 0:
            out[pos:pos + pause_samples] = 0.0
            pos += pause_samples

    return out


# =========================================================================
# 3. 断点续传
# =========================================================================


def get_checkpoint_dir(output_path: str) -> Path:
    """获取检查点目录（在输出文件旁建 .checkpoint 目录）。"""
    out = Path(output_path)
    ckpt_dir = out.parent / f".{out.name}_checkpoint"
    return ckpt_dir


def segment_path(ckpt_dir: Path, index: int) -> Path:
    """获取分段文件的确定性路径（无需 meta.json）。"""
    return ckpt_dir / f"seg_{index:04d}.wav"


def save_checkpoint_segment(ckpt_dir: Path, index: int, wav: np.ndarray, sr: int):
    """保存单个分段的音频到检查点目录。
    
    路径是确定性的 seg_{index:04d}.wav，无需额外 meta.json。
    相比原来每次读写全量 json 的 O(n²) 方案，此处仅一次 sf.write。
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(segment_path(ckpt_dir, index)), wav, sr)


def get_completed_indices(ckpt_dir: Path, total_segments: int = 0) -> Set[int]:
    """获取已完成的段落索引（基于文件系统扫描，无 json 读写开销）。
    
    使用 glob 扫描一次目录，返回有效的段落索引。
    Args:
        ckpt_dir: 检查点目录
        total_segments: 可选上限，避免扫描到无关文件
    Returns:
        已完成的段落索引集合
    """
    valid: Set[int] = set()
    try:
        for p in ckpt_dir.glob("seg_*.wav"):
            try:
                idx = int(p.stem.split("_")[1])  # "seg_0000" → 0
                if total_segments <= 0 or idx < total_segments:
                    valid.add(idx)
            except (ValueError, IndexError, OSError):
                continue
    except OSError:
        pass
    return valid


# =========================================================================
# 4. 主流程
# =========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS 长文本有声书生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 模型参数
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="HuggingFace 模型 ID 或本地路径 (默认: Qwen/Qwen3-TTS-12Hz-1.7B-Base)",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help='推理设备 (默认: mps; 可选: cpu, cuda:0)',
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="模型精度 (MPS 推荐 float16, CUDA 推荐 bfloat16)",
    )

    # 音色克隆参数
    parser.add_argument(
        "--ref-audio",
        default=None,
        dest="ref_audio",
        help="参考音频路径（用于音色克隆，3 秒以上效果更佳）",
    )
    parser.add_argument(
        "--ref-text",
        default=None,
        help="参考音频的文字内容（ICL 模式必填；x-vector 模式不需要）",
    )
    parser.add_argument(
        "--xvec",
        action="store_true",
        default=False,
        help="使用 x-vector only 模式（只需参考音频，不需要 ref_text，但音色相似度略低）",
    )

    # 参考音频历史
    parser.add_argument(
        "--ref-history",
        default=None,
        dest="ref_history",
        help="从历史记录加载参考音频（使用名称或 ID，详见 --list-refs）",
    )
    parser.add_argument(
        "--save-ref",
        default=None,
        dest="save_ref",
        help="将本次使用的参考音频保存到历史记录，并指定名称",
    )
    parser.add_argument(
        "--list-refs",
        action="store_true",
        default=False,
        dest="list_refs",
        help="列出所有保存的参考音频历史记录",
    )

    # 输入输出参数
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="输入文本文件路径（不指定则从标准输入读取）",
    )
    parser.add_argument(
        "--output", "-o",
        default="output.wav",
        help="输出音频文件路径 (默认: output.wav)",
    )
    parser.add_argument(
        "--language",
        default="Auto",
        help="合成语言 (默认: Auto 自动检测; 可选: Chinese, English, Japanese 等)",
    )

    # 分段参数
    parser.add_argument(
        "--max-seg-len",
        type=int,
        default=350,
        dest="max_seg_len",
        help="每段最大字符数 (默认: 350)",
    )

    # 生成参数
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        dest="max_new_tokens",
        help="每段最大生成 token 数 (默认: 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="采样温度 (默认: 0.9)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        dest="top_k",
        help="Top-k 采样参数 (默认: 50)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        dest="top_p",
        help="Top-p 采样参数 (默认: 1.0)",
    )

    # 韵律 / Subtalker 参数
    parser.add_argument(
        "--subtalker-temperature",
        type=float,
        default=0.9,
        dest="subtalker_temperature",
        help="Subtalker 韵律温度 (默认: 0.9; 调低=节奏更平, 调高=更多节奏变化)",
    )
    parser.add_argument(
        "--subtalker-top-k",
        type=int,
        default=50,
        dest="subtalker_top_k",
        help="Subtalker Top-K (默认: 50)",
    )
    parser.add_argument(
        "--subtalker-top-p",
        type=float,
        default=1.0,
        dest="subtalker_top_p",
        help="Subtalker Top-P (默认: 1.0)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        dest="repetition_penalty",
        help="重复惩罚系数 (默认: 1.05; 提高可增加韵律变化)",
    )

    # 音频后处理参数（默认 None 表示未指定，由交互式菜单处理）
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="语速倍率 (默认交互式选择; 0.9 稍慢; 1.0 原速; 0.8 更慢)",
    )
    parser.add_argument(
        "--segment-gap",
        type=float,
        default=None,
        dest="segment_gap",
        help="段间停顿秒数 (默认交互式选择; 1.5 秒推荐)",
    )
    parser.add_argument(
        "--breathing-pause",
        type=float,
        default=None,
        dest="breathing_pause",
        help="句间气口停顿秒数 (默认交互式选择; 0.25 推荐; 0 关闭)",
    )

    # 功能参数
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="启用断点续传（从上次中断处继续）",
    )
    parser.add_argument(
        "--keep-segments",
        action="store_true",
        default=False,
        dest="keep_segments",
        help="保留各分段的独立音频文件",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help="仅显示分段结果，不实际生成语音",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="减少输出信息",
    )

    # 并行生成参数
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=1,
        dest="batch_size",
        help=(
            "并行生成批大小。MPS/CUDA 设备上强制为 1，因 GPU 推理非线程安全；"
            "CPU 上可设为 2-4 提升速度 (默认: 1)"
        ),
    )

    # ASR 校验参数
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help=(
            "启用 ASR 校验：自动识别段头/段尾缺字的缺陷段并重新生成。"
            "需要安装 faster-whisper 或 openai-whisper。适合隔夜批量任务。"
        ),
    )
    parser.add_argument(
        "--verify-max-retries",
        type=int,
        default=3,
        dest="verify_max_retries",
        help="单段 ASR 校验最大重试次数 (默认: 3)",
    )

    return parser


# 模块级 dtype 映射，避免 parse_dtype 每次调用重新构建 dict
_DTYPE_MAP: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_dtype(s: str) -> torch.dtype:
    try:
        return _DTYPE_MAP[s]
    except KeyError:
        valid = ", ".join(_DTYPE_MAP.keys())
        print(f"❌ 无效精度 '{s}'，可选: {valid}", file=sys.stderr)
        sys.exit(1)


def _streaming_merge(
    seg_dir: Path,
    total_segments: int,
    completed: Set[int],
    sr: int,
    segment_gap: float,
    breathing_pause: float,
    segments_text: List[str],
    speed: float,
    output_path: str,
    quiet: bool,
) -> Tuple[int, float]:
    """流式合并：直接将各段写入输出文件，避免内存中保留所有音频数据。

    流程:
      1. 打开输出文件，逐段 stream-write（每段只占一个 ndarray 内存）
      2. 变速时重新读回做一次 O(n) 扫描
      3. 通过追踪 written_samples 精确计算时长，避免最终再读一次全文件

    注意：退出前自动清理 tmp_output 临时文件。
    """
    tmp_output = Path(output_path).with_suffix(".tmp.wav")
    written_samples = 0
    # 段间静音 buffer 缓存（复用同一数组，避免 N-1 次重复分配）
    gap_samples = int(segment_gap * sr) if segment_gap > 0 else 0
    gap_buf = np.zeros(gap_samples, dtype=np.float32) if gap_samples > 0 else None

    try:
        # ---- 第 1 步：流式写入临时文件 ----
        with sf.SoundFile(str(tmp_output), 'w', samplerate=sr, channels=1, format='WAV') as fout:
            for i in range(total_segments):
                if i not in completed:
                    continue
                seg_path = seg_dir / f"seg_{i:04d}.wav"
                if not seg_path.is_file():
                    continue

                wav, _ = sf.read(str(seg_path), dtype=np.float32)
                # 音频归一化：淡入 + 音量统一 → 消除开头爆音和忽大忽小
                wav = normalize_audio(wav, sr)
                # 气口停顿（insert_breathing_pauses 会改变 wav 长度）
                if breathing_pause > 0 and i < len(segments_text):
                    wav = insert_breathing_pauses(wav, segments_text[i], sr, breathing_pause)
                fout.write(wav)
                written_samples += len(wav)
                del wav

                # 段间静音（复用 gap_buf）
                if gap_buf is not None and i < total_segments - 1:
                    fout.write(gap_buf)
                    written_samples += gap_samples

            fout.flush()

        # ---- 第 2 步：输出最终文件 ----
        if abs(speed - 1.0) > 0.01:
            if not quiet:
                print(f"\n   🏎️ 变速: {speed}x...")
            data, _ = sf.read(str(tmp_output), dtype=np.float32)
            data = adjust_speed(data, sr, speed)
            sf.write(str(output_path), data, sr)
            total_dur = len(data) / sr
            del data
        else:
            os.replace(str(tmp_output), str(output_path))
            # 精确时长：从追踪的 written_samples 计算，无需重新读取文件
            total_dur = written_samples / sr

    except Exception:
        # 异常时清理临时文件
        if tmp_output.exists():
            tmp_output.unlink(missing_ok=True)
        raise
    else:
        # 正常完成后清理临时文件（变速路径：tmp_output 仍存在）
        if tmp_output.exists():
            tmp_output.unlink(missing_ok=True)

    gc.collect()
    return sr, total_dur  # 返回采样率和总时长


def post_process_interactive(
    seg_dir: Path,
    total_segments: int,
    completed: Set[int],
    sr: int,
    segments: List[str],
    output_path: str,
    quiet: bool,
    keep_segments: bool,
    cli_speed: Optional[float] = None,
    cli_segment_gap: Optional[float] = None,
    cli_breathing_pause: Optional[float] = None,
) -> Optional[Tuple[int, float]]:
    """生成完成后交互式后处理选择。

    自动检测是否在终端中运行。在非交互环境、安静模式、或提供了 CLI 后处理参数时，
    直接使用参数合并，不提示用户。

    Args:
        seg_dir: 分段音频目录
        total_segments: 总段数
        completed: 完成段索引集合
        sr: 采样率
        segments: 各段文本列表
        output_path: 输出文件路径
        quiet: 安静模式
        keep_segments: 保留分段文件
        cli_speed: CLI 提供的语速（None 表示未指定）
        cli_segment_gap: CLI 提供的段间停顿
        cli_breathing_pause: CLI 提供的气口停顿

    Returns:
        (sr, total_dur) 如果合并完成；None 如果跳过合并
    """
    # 默认后处理参数
    DEFAULT_SPEED = 0.9
    DEFAULT_GAP = 1.5
    DEFAULT_PAUSE = 0.25

    speed = cli_speed if cli_speed is not None else DEFAULT_SPEED
    segment_gap = cli_segment_gap if cli_segment_gap is not None else DEFAULT_GAP
    breathing_pause = cli_breathing_pause if cli_breathing_pause is not None else DEFAULT_PAUSE

    # 非交互式环境：直接使用参数合并
    if not sys.stdin.isatty() or quiet or cli_speed is not None or cli_segment_gap is not None or cli_breathing_pause is not None:
        sr_out, total_dur = _streaming_merge(
            seg_dir, total_segments, completed, sr,
            segment_gap, breathing_pause, segments, speed, output_path, quiet,
        )
        return sr_out, total_dur

    # 交互式菜单
    while True:
        print(f"\n{'='*50}")
        print(f"  ✅ 所有段落已生成完成！共 {len(completed)} 段")
        print(f"{'='*50}")
        print(f"  请选择后处理方式：")
        print(f"    [1] 直接合并输出（推荐参数：语速 {DEFAULT_SPEED}x，段间 {DEFAULT_GAP}s，气口 {DEFAULT_PAUSE}s）")
        print(f"    [2] 自定义后处理参数并合并")
        print(f"    [3] 跳过合并")
        if keep_segments:
            print(f"         (--keep-segments 已启用，分段文件将保存到独立目录)")
        print(f"{'='*50}")
        choice = input("  请输入选择 (1/2/3): ").strip()

        if choice == '1':
            sr_out, total_dur = _streaming_merge(
                seg_dir, total_segments, completed, sr,
                DEFAULT_GAP, DEFAULT_PAUSE, segments, DEFAULT_SPEED, output_path, quiet,
            )
            print(f"\n  ✅ 合并完成！输出文件: {output_path}")
            return sr_out, total_dur

        elif choice == '2':
            try:
                print("\n  自定义后处理参数（直接回车使用括号中的默认值）：")
                s = input(f"  语速倍率 [{speed}]: ").strip()
                if s:
                    speed = float(s)
                g = input(f"  段间停顿(秒) [{segment_gap}]: ").strip()
                if g:
                    segment_gap = float(g)
                p = input(f"  气口停顿(秒) [{breathing_pause}]: ").strip()
                if p:
                    breathing_pause = float(p)

                confirm = input(f"\n  确认合并？(语速 {speed}x，段间 {segment_gap}s，气口 {breathing_pause}s) [Y/n]: ").strip().lower()
                if confirm in ('', 'y', 'yes'):
                    sr_out, total_dur = _streaming_merge(
                        seg_dir, total_segments, completed, sr,
                        segment_gap, breathing_pause, segments, speed, output_path, quiet,
                    )
                    print(f"\n  ✅ 合并完成！输出文件: {output_path}")
                    return sr_out, total_dur
                else:
                    print("  ⏭️  已取消，返回菜单")
                    continue
            except ValueError:
                print("  ❌ 无效输入，请输入数字")
                continue

        elif choice == '3':
            print("  ✅ 已跳过合并")
            return None

        else:
            print("  ❌ 无效选择，请输入 1-3")


# 全局中断标志，用于信号处理
_interrupted = False


def _signal_handler(signum, frame):
    global _interrupted
    if not _interrupted:
        print("\n⚠️  接收到中断信号，正在停止（当前段完成后不再继续）...", file=sys.stderr)
        _interrupted = True
    else:
        print("\n❌ 再次按 Ctrl+C 强制退出", file=sys.stderr)
        sys.exit(1)


def main():
    global _interrupted
    parser = build_parser()
    args = parser.parse_args()

    # 注册 Ctrl+C 信号处理器
    signal.signal(signal.SIGINT, _signal_handler)

    # ------------------------------------------------------------------
    # 处理 --list-refs（列出历史记录后立即退出）
    # ------------------------------------------------------------------
    if args.list_refs:
        entries = ref_history.load_all()
        if not entries:
            print("📭 暂无保存的参考音频历史记录")
        else:
            print(f"\n📚 参考音频历史记录 ({len(entries)} 条):")
            print("=" * 50)
            for e in entries:
                xvec_str = " [x-vector]" if e.get("xvec") else ""
                print(f"   [{e['id']}] {e['name']}{xvec_str}")
                print(f"       音频: {e['audio']}")
                print(f"       文本: {e['text'][:80]}{'...' if len(e['text']) > 80 else ''}")
                print(f"       创建: {e['created']}")
            print("=" * 50)
        sys.exit(0)

    # ------------------------------------------------------------------
    # 处理 --ref-history（从历史记录加载参考音频）
    # ------------------------------------------------------------------
    if args.ref_history:
        entry = ref_history.find(args.ref_history)
        if entry is None:
            print(f"❌ 未找到历史记录: {args.ref_history}", file=sys.stderr)
            print("   使用 --list-refs 查看可用记录", file=sys.stderr)
            sys.exit(1)
        if not args.quiet:
            print(f"📂 从历史加载参考: {entry['name']} ({entry['created']})")
        args.ref_audio = entry["audio"]
        args.ref_text = entry.get("text", "")
        if entry.get("xvec"):
            args.xvec = True

    # 验证参考音频
    if not args.ref_audio:
        print("❌ 错误：需要提供 --ref-audio（音频路径）或 --ref-history（历史记录名称）",
              file=sys.stderr)
        print("   使用 --list-refs 可查看已有的历史记录", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 读取输入文本
    # ------------------------------------------------------------------
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        if not args.quiet:
            print("📖 从标准输入读取文本（Ctrl+D 结束）...", file=sys.stderr)
        text = sys.stdin.read()

    text = text.strip()
    if not text:
        print("❌ 错误：输入文本为空", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 文本分段
    # ------------------------------------------------------------------
    if not args.quiet:
        print(f"📄 输入文本长度: {len(text)} 字符")

    segments = segment_text(text, max_chars=args.max_seg_len)

    if not args.quiet:
        print(f"✂️  分段数量: {len(segments)}")
        for i, seg in enumerate(segments):
            print(f"   段 [{i+1}/{len(segments)}]: {len(seg)} 字 — {seg[:60]}...")

    if args.dry_run:
        print("\n✅ 干运行完成。未实际生成语音。")
        return

    # ------------------------------------------------------------------
    # 加载模型
    # ------------------------------------------------------------------
    if not args.quiet:
        print(f"\n🔄 加载模型: {args.model}")
        print(f"   设备: {args.device}, 精度: {args.dtype}")

    device = args.device
    dtype = parse_dtype(args.dtype)

    # MPS 不支持 bfloat16
    if device == "mps" and dtype == torch.bfloat16:
        if not args.quiet:
            print("   ⚠️  MPS 不支持 bfloat16，切换为 float32")
        dtype = torch.float32

    # Flash attention: CUDA 可用时启用，MPS 不支持
    attn_impl = None
    if device.startswith("cuda"):
        attn_impl = "flash_attention_2"

    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_impl,
        local_files_only=True,
    )
    if not args.quiet:
        print(f"   ✅ 模型加载完成 ({time.time() - t0:.1f}s)")

    # MPS 内存优化：加载完成后清理缓存，释放加载过程的临时张量
    if device == "mps":
        torch.mps.empty_cache()

    # 强制 batch_size 兼容性检查
    if args.batch_size > 1 and device != "cpu":
        if not args.quiet:
            print(f"   ⚠️  {device} 设备不支持并行批处理，强制 batch=1")
        args.batch_size = 1

    # ------------------------------------------------------------------
    # 创建音色克隆 Prompt
    # ------------------------------------------------------------------
    if not args.quiet:
        print(f"\n🎤 创建音色克隆...")

    # 加载参考音频
    ref_audio_path = args.ref_audio
    ref_text = args.ref_text

    if args.xvec:
        prompt_items = tts.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            x_vector_only_mode=True,
        )
    else:
        if not ref_text:
            print("❌ 错误：ICL 模式需要提供 --ref_text（或使用 --xvec 模式）", file=sys.stderr)
            sys.exit(1)
        prompt_items = tts.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )

    if not args.quiet:
        print(f"   ✅ 音色克隆完成")

    # 保存到历史记录（如果指定了 --save-ref）
    if args.save_ref:
        entry = ref_history.add(
            name=args.save_ref,
            audio_src=ref_audio_path,
            text=ref_text or "",
            xvec=args.xvec,
        )
        if not args.quiet:
            print(f"   💾 已保存到参考音频历史: {entry['name']} ({entry['created']})")

    # ------------------------------------------------------------------
    # ASR 校验初始化（如果启用了 --verify）
    # ------------------------------------------------------------------
    if args.verify:
        if not args.quiet:
            print("")
        asr_ok = init_asr(quiet=args.quiet)
        if not asr_ok:
            print("⚠️  --verify 已启用但 ASR 不可用，将跳过校验", file=sys.stderr)
            args.verify = False

    # ------------------------------------------------------------------
    # 逐段生成语音
    # ------------------------------------------------------------------
    ckpt_dir = get_checkpoint_dir(args.output) if args.resume else None
    # 非 resume 模式也创建临时目录保存分段，避免全部驻留内存
    streaming_dir = ckpt_dir
    cleanup_streaming = False
    if streaming_dir is None:
        streaming_dir = Path(tempfile.mkdtemp(prefix="qwen_tts_stream_"))
        cleanup_streaming = True

    completed: Set[int] = set()
    if args.resume and ckpt_dir and ckpt_dir.exists():
        completed = get_completed_indices(ckpt_dir)
        if completed and not args.quiet:
            print(f"\n🔄 断点续传：已完成的段落: {sorted(completed)}")

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        subtalker_dosample=True,
        subtalker_top_k=args.subtalker_top_k,
        subtalker_top_p=args.subtalker_top_p,
        subtalker_temperature=args.subtalker_temperature,
    )

    if not args.quiet:
        print(f"\n🔊 开始逐段合成 ({len(segments)} 段)...")

    language = args.language
    total_time = 0.0
    segment_sr: Optional[int] = None
    segment_count = 0
    # 在循环中追踪，避免后续 O(n) 文件系统扫描
    all_completed: Set[int] = set(completed)

    # ASR 校验统计
    verify_stats = {"checked": 0, "passed": 0, "retried": 0, "failed": 0, "skipped_short": 0}

    # MPS/CUDA 批量兼容：GPU 推理非线程安全，强制 batch=1
    batch_size = args.batch_size
    if batch_size > 1 and device != "cpu":
        if not args.quiet:
            print(f"   ⚠️  {device} 不支持并行生成，强制 batch=1")
        batch_size = 1
    if batch_size > 1 and not args.quiet:
        print(f"   ⚡ 并行模式: batch_size={batch_size}")

    def _gen_one(idx: int, seg_text: str) -> Tuple[int, np.ndarray, int, float]:
        t0 = time.time()
        wavs, sr = tts.generate_voice_clone(
            text=seg_text,
            language=language,
            voice_clone_prompt=prompt_items,
            **gen_kwargs,
        )
        elapsed = time.time() - t0
        # MPS: 每段生成后清理缓存，防止多段累积 OOM
        if device == "mps":
            torch.mps.empty_cache()
        return idx, wavs[0], sr, elapsed

    # 异步检查点保存器（单独的线程池，不阻塞生成主线程）
    ckpt_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    if ckpt_dir is not None:
        ckpt_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    if batch_size > 1:
        # ======== 并行路径 (CPU only) ========
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as pool:
            seg_map: dict = {}
            for i, seg_text in enumerate(segments):
                if _interrupted:
                    break
                seg_path = streaming_dir / f"seg_{i:04d}.wav"
                if i in completed:
                    all_completed.add(i)
                    segment_count += 1
                    seg_wav, sr = sf.read(str(seg_path))
                    segment_sr = sr
                    if not args.quiet:
                        seg_dur = len(seg_wav) / sr
                        print(f"   ⏩ 段 [{i+1}/{len(segments)}] 已存在 ({seg_dur:.1f}s)")
                    continue
                future = pool.submit(_gen_one, i, seg_text)
                seg_map[future] = (i, seg_text)

            for future in concurrent.futures.as_completed(seg_map):
                i, seg_text = seg_map[future]
                try:
                    _, seg_wav, sr, elapsed = future.result()
                    segment_sr = sr
                    seg_path = streaming_dir / f"seg_{i:04d}.wav"
                    sf.write(str(seg_path), seg_wav, sr)
                    all_completed.add(i)
                    segment_count += 1
                    total_time += elapsed
                    # 异步保存检查点
                    if ckpt_executor is not None:
                        ckpt_executor.submit(save_checkpoint_segment, ckpt_dir, i, seg_wav, sr)
                    if not args.quiet:
                        seg_dur = len(seg_wav) / sr
                        print(f"   ✅ 段 [{i+1}/{len(segments)}] ({len(seg_text)}字) → {seg_dur:.1f}s ({elapsed:.1f}s)")
                except Exception as e:
                    print(f"\n❌ 段 [{i+1}] 生成失败: {e}", file=sys.stderr)
                    print(f"   文本: {seg_text[:100]}...", file=sys.stderr)
                    error_log = Path(args.output).parent / f"error_seg_{i+1}.txt"
                    error_log.write_text(seg_text, encoding="utf-8")
                    print(f"   失败文本已保存到: {error_log}", file=sys.stderr)
                    continue
    else:
        # ======== 串行路径 (MPS/CUDA/CPU batch=1) ========
        for i, seg_text in enumerate(segments):
            if _interrupted:
                break

            seg_path = streaming_dir / f"seg_{i:04d}.wav"

            if i in completed:
                if not args.quiet:
                    print(f"   ⏩ 段 [{i+1}/{len(segments)}] 已存在，跳过")
                all_completed.add(i)
                segment_count += 1
                continue

            if not args.quiet:
                print(f"   🎯 段 [{i+1}/{len(segments)}] ({len(seg_text)} 字)...", end=" ", flush=True)

            # ASR 校验 + 自动重试
            max_retry = args.verify_max_retries if args.verify else 0
            best_wav = None
            best_sr = None
            best_elapsed = 0.0
            success = False
            # 重试使用本地副本，不修改共享 gen_kwargs
            retry_kwargs = dict(gen_kwargs)

            for attempt in range(max_retry + 1):
                if _interrupted:
                    break
                try:
                    # 用副本调用 _gen_one（_gen_one 内部引用的 gen_kwargs 是外层闭包）
                    # 这里重新构造调用以避免修改全局 gen_kwargs
                    t0 = time.time()
                    wavs, sr_local = tts.generate_voice_clone(
                        text=seg_text,
                        language=language,
                        voice_clone_prompt=prompt_items,
                        **retry_kwargs,
                    )
                    elapsed = time.time() - t0
                    if device == "mps":
                        torch.mps.empty_cache()
                    seg_wav = wavs[0]

                    # 流式写入文件
                    sf.write(str(seg_path), seg_wav, sr_local)

                    if attempt == 0:
                        total_time += elapsed
                        segment_sr = sr_local
                        best_wav, best_sr, best_elapsed = seg_wav, sr_local, elapsed
                    else:
                        verify_stats["retried"] += 1

                    segment_sr = sr_local

                    if args.verify:
                        verify_stats["checked"] += 1
                        passed, asr_text, asr_conf, fail_reason = verify_segment_head_tail(
                            str(seg_path), seg_text, language=language,
                        )
                        if passed:
                            verify_stats["passed"] += 1
                            success = True
                            if not args.quiet:
                                asr_info = f" (ASR✓ {asr_conf:.0%})" if asr_conf > 0 else ""
                                print(f"✅ 校验通过{asr_info}", end="")
                            break
                        else:
                            if not args.quiet:
                                print(f"⚠️ 校验失败: {fail_reason} (重试 {attempt+1}/{max_retry})", flush=True)
                            if attempt < max_retry:
                                # 微调 temperature 增加随机性（仅本地副本，不影响后续段落）
                                retry_kwargs["temperature"] = min(
                                    retry_kwargs["temperature"] + 0.1, 1.2
                                )
                                if not args.quiet:
                                    print(f"      ↻ 重新生成 (temp={retry_kwargs['temperature']:.2f})...", end=" ", flush=True)
                    else:
                        # 未启用校验 → 一次成功
                        success = True
                        break

                except Exception as e:
                    print(f"\n❌ 段 [{i+1}] 生成失败: {e}", file=sys.stderr)
                    print(f"   文本: {seg_text[:100]}...", file=sys.stderr)
                    error_log = Path(args.output).parent / f"error_seg_{i+1}.txt"
                    error_log.write_text(seg_text, encoding="utf-8")
                    print(f"   失败文本已保存到: {error_log}", file=sys.stderr)
                    break  # 跳出 retry 循环，不再尝试

            # ---- 处理最终结果 ----
            if _interrupted:
                break

            if best_wav is not None:
                # 用最后一次成功或最佳尝试的结果写入最终文件
                if best_wav is not seg_wav or best_sr != sr:
                    sf.write(str(seg_path), best_wav, best_sr)
                all_completed.add(i)
                segment_count += 1

                # 异步保存检查点
                if ckpt_executor is not None:
                    ckpt_executor.submit(save_checkpoint_segment, ckpt_dir, i, best_wav, best_sr)

                seg_dur = len(best_wav) / best_sr if best_wav is not None else 0
                if not args.quiet:
                    if success:
                        eta_remaining = (total_time / segment_count) * (len(segments) - i - 1)
                        print(f" {seg_dur:.1f}s (ETA {eta_remaining:.0f}s)")
                    else:
                        if args.verify:
                            verify_stats["failed"] += 1
                            print(f"   ⚠️ 校验失败 (已保留当前版本)")
                        else:
                            print(f" {seg_dur:.1f}s")
            elif not args.quiet:
                print(f"   ❌ 段 [{i+1}] 生成失败")

    # 等待所有异步检查点写入完成
    if ckpt_executor is not None:
        ckpt_executor.shutdown(wait=True)

    if segment_count == 0:
        print("❌ 错误：没有成功生成任何语音段", file=sys.stderr)
        sys.exit(1)

    # 确保 segment_sr 已设置
    if segment_sr is None:
        print("❌ 错误：未能获取采样率信息", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 后处理（交互式选择 — 生成完成后让用户选择如何处理）
    # ------------------------------------------------------------------
    result = post_process_interactive(
        seg_dir=streaming_dir,
        total_segments=len(segments),
        completed=all_completed,
        sr=segment_sr,
        segments=segments,
        output_path=args.output,
        quiet=args.quiet,
        keep_segments=args.keep_segments,
        cli_speed=args.speed,
        cli_segment_gap=args.segment_gap,
        cli_breathing_pause=args.breathing_pause,
    )

    gc.collect()

    if result is not None:
        sr, total_dur = result

        # 可选：保存各段独立文件
        if args.keep_segments:
            seg_dir = Path(args.output).parent / f"{Path(args.output).stem}_segments"
            seg_dir.mkdir(parents=True, exist_ok=True)
            for i in all_completed:
                src = streaming_dir / f"seg_{i:04d}.wav"
                dst = seg_dir / f"seg_{i+1:04d}.wav"
                if src.is_file():
                    shutil.copy2(str(src), str(dst))
            if not args.quiet:
                print(f"   📁 独立分段已保存到: {seg_dir}")

        # 清理临时目录
        if cleanup_streaming and streaming_dir.exists():
            shutil.rmtree(streaming_dir)
        elif ckpt_dir and ckpt_dir.exists() and not args.keep_segments:
            shutil.rmtree(ckpt_dir)

        # ASR 校验统计
        if args.verify and verify_stats["checked"] > 0:
            pct = verify_stats["passed"] / max(verify_stats["checked"], 1) * 100
            lines = [
                f"\n   🎤 ASR 校验报告:",
                f"      校验段数: {verify_stats['checked']}",
                f"      一次通过: {verify_stats['passed']} ({pct:.0f}%)",
                f"      重试修正: {verify_stats['retried']}",
            ]
            if verify_stats["failed"] > 0:
                lines.append(f"      校验失败: {verify_stats['failed']}（已保留，建议人工复查）")
            print("\n".join(lines))

        # 输出统计
        if not args.quiet:
            print(f"\n{'='*40}")
            print(f"✅ 合成完成!")
            print(f"   输出文件: {args.output}")
            print(f"   总时长: {total_dur:.1f}s ({total_dur/60:.1f} 分钟)")
            print(f"   采样率: {sr} Hz")
            print(f"   总段数: {segment_count}")
            print(f"   生成耗时: {total_time:.1f}s")
            print(f"   实时率: {total_time/total_dur:.2f}x")
            print(f"{'='*40}")
        else:
            print(f"✅ 合成完成: {args.output} ({total_dur:.1f}s)")
    else:
        # 用户选择跳过合并
        if not args.quiet:
            print(f"\n{'='*40}")
            print(f"✅ 段落生成完成（未合并）")

            # ASR 校验统计
            if args.verify and verify_stats["checked"] > 0:
                pct = verify_stats["passed"] / max(verify_stats["checked"], 1) * 100
                print(f"\n   🎤 ASR 校验报告:")
                print(f"      校验段数: {verify_stats['checked']}")
                print(f"      一次通过: {verify_stats['passed']} ({pct:.0f}%)")
                print(f"      重试修正: {verify_stats['retried']}")
                if verify_stats["failed"] > 0:
                    print(f"      校验失败: {verify_stats['failed']}（已保留，建议人工复查）")

            print(f"   总段数: {segment_count}")
            print(f"   分段文件位于: {streaming_dir}")
            print(f"   生成耗时: {total_time:.1f}s")
            if args.keep_segments:
                print(f"   --keep-segments 已启用，未清理分段文件")
            print(f"{'='*40}")


if __name__ == "__main__":
    main()
