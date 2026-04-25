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
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

# =========================================================================
# 1. 文本分段
# =========================================================================

# 次级分隔符（用于超长句的强制拆分）
SUB_SEPARATOR = re.compile(r"([，、,、\s]{1,2})")


def split_sentences(text: str) -> List[str]:
    """将文本按句子边界切分，保留标点符号在句尾。

    使用逐字符扫描方式，在句子结束标点处切分：
      - 中文：。！？
      - 英文：.!?
      - 换行：\\n
    """
    sentences = []
    buf = []
    for ch in text:
        buf.append(ch)
        if ch in "。！？.!?\n":
            s = "".join(buf).strip()
            if s:
                sentences.append(s)
            buf = []
    # 剩余部分
    remaining = "".join(buf).strip()
    if remaining:
        # 如果剩余太短且已有句子，合并到前一句
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

    Args:
        text: 输入文本
        max_chars: 每段最大字符数

    Returns:
        分段后的文本列表
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    segments = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # 单句超长：先用次级分隔符（逗号/空格）拆分，如果拆不动（无分隔符）则按字符数强制切割
        if len(sent) > max_chars:
            # 先把当前累积的段落提交
            if current.strip():
                segments.append(current.strip())
                current = ""
            # 尝试用逗号/空格拆分
            sub_sentences = re.split(r"([，、,、\s]{1,2})", sent)
            # 如果 regex 拆不动（整个句子是连续的），按字符数强制切分
            if len(sub_sentences) <= 1:
                for i in range(0, len(sent), max_chars):
                    segments.append(sent[i:i + max_chars])
                continue
            sub_buf = ""
            for part in sub_sentences:
                if len(sub_buf) + len(part) > max_chars and sub_buf.strip():
                    segments.append(sub_buf.strip())
                    sub_buf = part
                else:
                    sub_buf += part
            if sub_buf.strip():
                segments.append(sub_buf.strip())
            continue

        # 正常情况：累积到 max_chars 就切分
        if len(current) + len(sent) > max_chars and current.strip():
            segments.append(current.strip())
            current = sent
        else:
            current += sent

    # 最后一段
    if current.strip():
        segments.append(current.strip())

    return segments


# =========================================================================
# 2. 音频处理（变速 + 停顿 + 合并）
# =========================================================================


def adjust_speed(wav: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """调整语速，不变调。

    Args:
        wav: 音频数据
        sr: 采样率
        speed: 速度倍率 (0.5=慢一倍, 1.0=原速, 1.5=快50%)

    Returns:
        变速后的音频
    """
    if abs(speed - 1.0) < 0.01:
        return wav
    return librosa.effects.time_stretch(y=wav.astype(np.float32), rate=speed)


def make_silence(duration_s: float, sr: int) -> np.ndarray:
    """生成一段静音。"""
    n = int(duration_s * sr)
    return np.zeros(n, dtype=np.float32)


def merge_audio_segments(
    segments: List[Tuple[np.ndarray, int]],
    segment_gap: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """将多个音频段合并为一个，段间可插入静音间隔。

    Args:
        segments: [(wav_array, sample_rate), ...]
        segment_gap: 段间静音间隔（秒）

    Returns:
        (merged_wav, sample_rate)
    """
    if not segments:
        raise ValueError("没有音频段可合并")

    sr = segments[0][1]
    parts = []
    for i, (wav, _) in enumerate(segments):
        if i > 0 and segment_gap > 0:
            parts.append(make_silence(segment_gap, sr))
        parts.append(wav)
    merged = np.concatenate(parts)
    return merged, sr

def insert_breathing_pauses(
    wav: np.ndarray,
    text: str,
    sr: int,
    pause_s: float = 0.3,
) -> np.ndarray:
    """在句子边界插入短停顿，模拟气口。

    根据文本中的句子结束标点，在音频对应位置插入停顿。
    采用按比例估算位置的方式（非精准对齐）。

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

    # 按文本长度比例估算每个句子在音频中的起止位置
    total_chars = len(text)
    if total_chars == 0:
        return wav

    # 将 sentences 恢复为带标点的完整形式（split_sentences 已经保留标点）
    parts = []
    current_char = 0
    for sent in sentences:
        start_ratio = current_char / total_chars
        end_ratio = (current_char + len(sent)) / total_chars
        start_sample = int(start_ratio * len(wav))
        end_sample = int(end_ratio * len(wav))
        current_char += len(sent)

        seg = wav[start_sample:end_sample]
        parts.append(seg)
        pause_samples = int(pause_s * sr)
        parts.append(np.zeros(pause_samples, dtype=np.float32))

    if parts:
        # 去掉最后一个多余的停顿
        return np.concatenate(parts[:-1])
    return wav


# =========================================================================
# 3. 断点续传
# =========================================================================


def get_checkpoint_dir(output_path: str) -> Path:
    """获取检查点目录（在输出文件旁建 .checkpoint 目录）。"""
    out = Path(output_path)
    ckpt_dir = out.parent / f".{out.name}_checkpoint"
    return ckpt_dir


def save_checkpoint_segment(ckpt_dir: Path, index: int, wav: np.ndarray, sr: int):
    """保存单个分段的音频到检查点目录。"""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    seg_path = ckpt_dir / f"seg_{index:04d}.wav"
    sf.write(str(seg_path), wav, sr)
    # 同时保存元数据
    meta_path = ckpt_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    meta[str(index)] = str(seg_path)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def load_checkpoint(ckpt_dir: Path) -> dict:
    """加载检查点信息。"""
    meta_path = ckpt_dir / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def get_completed_indices(ckpt_dir: Path) -> set:
    """获取已完成的段落索引。"""
    meta = load_checkpoint(ckpt_dir)
    return set(int(k) for k in meta.keys())


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
        required=True,
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

    # 音频后处理参数
    parser.add_argument(
        "--speed",
        type=float,
        default=0.9,
        help="语速倍率 (默认: 0.9 稍慢; 1.0 原速; 0.8 更慢)",
    )
    parser.add_argument(
        "--segment-gap",
        type=float,
        default=1.5,
        dest="segment_gap",
        help="段间停顿秒数 (默认: 1.5 秒)",
    )
    parser.add_argument(
        "--breathing-pause",
        type=float,
        default=0.25,
        dest="breathing_pause",
        help="句间气口停顿秒数 (默认: 0.25; 0 关闭)",
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

    return parser


def parse_dtype(s: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[s]


def main():
    parser = build_parser()
    args = parser.parse_args()

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

    # ------------------------------------------------------------------
    # 逐段生成语音
    # ------------------------------------------------------------------
    ckpt_dir = get_checkpoint_dir(args.output) if args.resume else None
    completed = set()
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
    all_wavs: List[np.ndarray] = []
    total_time = 0.0

    for i, seg_text in enumerate(segments):
        if i in completed:
            # 从检查点加载已合成的音频
            meta = load_checkpoint(ckpt_dir)
            wav, sr = sf.read(meta[str(i)])
            all_wavs.append(wav)
            if not args.quiet:
                print(f"   ⏩ 段 [{i+1}/{len(segments)}] 已存在，跳过")
            continue

        if not args.quiet:
            print(f"   🎯 段 [{i+1}/{len(segments)}] ({len(seg_text)} 字)...", end=" ", flush=True)

        t_start = time.time()
        try:
            wavs, sr = tts.generate_voice_clone(
                text=seg_text,
                language=language,
                voice_clone_prompt=prompt_items,
                **gen_kwargs,
            )
            elapsed = time.time() - t_start
            total_time += elapsed

            seg_wav = wavs[0]
            all_wavs.append(seg_wav)

            # 保存检查点
            if ckpt_dir is not None:
                save_checkpoint_segment(ckpt_dir, i, seg_wav, sr)

            if not args.quiet:
                seg_dur = len(seg_wav) / sr
                print(f"✅ {seg_dur:.1f}s 音频 ({elapsed:.1f}s 生成)")

        except Exception as e:
            print(f"\n❌ 段 [{i+1}] 生成失败: {e}", file=sys.stderr)
            print(f"   文本: {seg_text[:100]}...", file=sys.stderr)
            # 保存失败段信息以便排查
            error_log = Path(args.output).parent / f"error_seg_{i+1}.txt"
            error_log.write_text(seg_text, encoding="utf-8")
            print(f"   失败文本已保存到: {error_log}", file=sys.stderr)
            continue

    if not all_wavs:
        print("❌ 错误：没有成功生成任何语音段", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 音频后处理：气口 + 段间停顿 + 变速
    # ------------------------------------------------------------------
    if not args.quiet:
        print(f"\n🔊 后处理: 语速={args.speed}x, 段间停顿={args.segment_gap}s, 气口={args.breathing_pause}s")

    # 1) 句间插入气口停顿
    if args.breathing_pause > 0:
        if not args.quiet:
            print(f"   插入气口停顿...")
        processed_wavs = []
        for i, (wav, seg_text) in enumerate(zip(all_wavs, segments)):
            wav_with_pause = insert_breathing_pauses(
                wav, seg_text, sr, pause_s=args.breathing_pause,
            )
            processed_wavs.append((wav_with_pause, sr))
    else:
        processed_wavs = [(w, sr) for w in all_wavs]

    # 2) 合并 + 段间停顿
    if not args.quiet:
        print(f"   合并音频 (段间停顿 {args.segment_gap}s)...")
    merged_wav, sr = merge_audio_segments(processed_wavs, segment_gap=args.segment_gap)

    # 3) 变速
    if abs(args.speed - 1.0) > 0.01:
        if not args.quiet:
            print(f"   语速调整: {args.speed}x...")
        merged_wav = adjust_speed(merged_wav, sr, speed=args.speed)

    # 保存输出
    sf.write(args.output, merged_wav, sr)

    # 可选：保存各段独立文件
    if args.keep_segments:
        seg_dir = Path(args.output).parent / f"{Path(args.output).stem}_segments"
        seg_dir.mkdir(parents=True, exist_ok=True)
        for i, wav in enumerate(all_wavs):
            sf.write(str(seg_dir / f"seg_{i+1:04d}.wav"), wav, sr)
        if not args.quiet:
            print(f"   📁 独立分段已保存到: {seg_dir}")

    # 清理检查点
    if ckpt_dir and ckpt_dir.exists() and not args.keep_segments:
        import shutil
        shutil.rmtree(ckpt_dir)

    # ------------------------------------------------------------------
    # 输出统计
    # ------------------------------------------------------------------
    total_duration = len(merged_wav) / sr
    if not args.quiet:
        print(f"\n{'='*40}")
        print(f"✅ 合成完成!")
        print(f"   输出文件: {args.output}")
        print(f"   总时长: {total_duration:.1f}s ({total_duration/60:.1f} 分钟)")
        print(f"   采样率: {sr} Hz")
        print(f"   总段数: {len(all_wavs)}")
        print(f"   生成耗时: {total_time:.1f}s")
        print(f"   实时率: {total_time/total_duration:.2f}x")
        print(f"{'='*40}")
    else:
        print(f"✅ 合成完成: {args.output} ({total_duration:.1f}s)")


if __name__ == "__main__":
    main()
