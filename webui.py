#!/usr/bin/env python3
"""
Qwen3-TTS 有声书生成器 - WebUI (Gradio)
==========================================

基于 Qwen3-TTS (Apache 2.0) — 阿里通义千问团队
  GitHub: https://github.com/QwenLM/Qwen3-TTS
  论文: https://arxiv.org/abs/2601.15621
  PyPI: qwen-tts

启动:
  source venv/bin/activate
  python webui.py
  # 浏览器打开 http://127.0.0.1:7860
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch

from long_tts import Qwen3TTSModel, segment_text, merge_audio_segments
from long_tts import adjust_speed, insert_breathing_pauses, split_sentences


# =====================================================================
# 全局状态
# =====================================================================

class AppState:
    def __init__(self):
        self.tts: Optional[Qwen3TTSModel] = None
        self.prompt_items = None
        self.model_loaded = False
        self.model_path = ""
        self.sample_rate = 24000


state = AppState()


def _resolve_local_cache(repo_id: str) -> Optional[str]:
    """Resolve a HuggingFace repo ID to a local cache snapshot path if cached."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir_name = "models--" + repo_id.replace("/", "--")
    repo_dir = cache_dir / repo_dir_name
    if not repo_dir.exists():
        return None
    refs_file = repo_dir / "refs" / "main"
    if not refs_file.exists():
        return None
    commit_hash = refs_file.read_text().strip()
    snapshot_dir = repo_dir / "snapshots" / commit_hash
    if snapshot_dir.exists():
        return str(snapshot_dir)
    return None


# =====================================================================
# 核心逻辑
# =====================================================================

def load_model(model_path: str, device: str, dtype_str: str) -> str:
    global state
    if state.model_loaded and state.model_path == model_path:
        return f"✅ 模型已加载: {model_path}"

    # 自动解析 repo ID 到本地缓存路径，避免 huggingface_hub 网络重试
    resolved = _resolve_local_cache(model_path) or model_path
    if resolved != model_path:
        print(f"[load_model] 自动解析到本地缓存: {resolved}")

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(dtype_str, torch.float32)
    if device == "mps" and dtype == torch.bfloat16:
        dtype = torch.float32
    attn_impl = "flash_attention_2" if device.startswith("cuda") else None

    try:
        t0 = time.time()
        state.tts = Qwen3TTSModel.from_pretrained(
            resolved,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
            local_files_only=True,
        )

        state.model_loaded = True
        state.model_path = model_path
        return f"✅ 模型加载完成 ({time.time() - t0:.1f}s)"
    except Exception as e:
        state.model_loaded = False
        return f"❌ 模型加载失败: {e}"


def unload_model() -> str:
    global state
    if state.tts is None:
        return "⏳ 模型未加载"
    try:
        state.tts = None
        state.prompt_items = None
        state.model_loaded = False
        state.model_path = ""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "✅ 模型已从内存卸载"
    except Exception as e:
        return f"❌ 卸载失败: {e}"


def create_voice_clone(audio_path: str, ref_text: str, use_xvec: bool) -> str:
    global state
    if not state.model_loaded or state.tts is None:
        return "❌ 请先加载模型"
    if not audio_path:
        return "❌ 请上传参考音频"
    try:
        if use_xvec:
            state.prompt_items = state.tts.create_voice_clone_prompt(
                ref_audio=audio_path, x_vector_only_mode=True,
            )
        else:
            if not ref_text or not ref_text.strip():
                return "❌ ICL 模式需要提供参考音频的文字内容"
            state.prompt_items = state.tts.create_voice_clone_prompt(
                ref_audio=audio_path, ref_text=ref_text.strip(), x_vector_only_mode=False,
            )
        return "✅ 音色克隆完成"
    except Exception as e:
        state.prompt_items = None
        return f"❌ 音色克隆失败: {e}"


def analyze_voice_profile(audio_path: str) -> str:
    """分析参考音频，提取声音特征并推荐性格描述词语。"""
    if not audio_path:
        return "⚠️ 请先上传参考音频"

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        # ---- 1. 音高分析 ----
        f0, voiced_flag, _ = librosa.pyin(y, fmin=65, fmax=2093, sr=sr)
        pitched_f0 = f0[voiced_flag] if voiced_flag.any() else np.array([])
        mean_pitch = np.mean(pitched_f0) if len(pitched_f0) > 0 else 0
        pitch_std = np.std(pitched_f0) if len(pitched_f0) > 0 else 0

        # ---- 2. 能量/音量 ----
        rms = librosa.feature.rms(y=y)[0]
        mean_vol = float(np.mean(rms))
        vol_std = float(np.std(rms))
        vol_range = float(np.max(rms) - np.min(rms))

        # ---- 3. 语速（用VAD估计） ----
        # 简单VAD：能量阈值
        threshold = np.mean(rms) * 0.3
        is_speech = rms > threshold
        speech_frames = np.sum(is_speech)
        total_frames = len(rms)
        speech_ratio = speech_frames / max(total_frames, 1)
        duration = len(y) / sr

        # ---- 4. 生成声音画像 ----
        lines = []
        lines.append("📊 声音特征分析")
        lines.append("=" * 40)

        # 音高描述
        if mean_pitch > 0:
            if mean_pitch > 220:
                pitch_desc = "较高"
                pitch_tags = ["明亮", "清亮"]
            elif mean_pitch > 160:
                pitch_desc = "中等"
                pitch_tags = ["自然", "平和"]
            else:
                pitch_desc = "较低"
                pitch_tags = ["沉稳", "低沉", "浑厚"]
            lines.append(f"🎵 音高: {pitch_desc} ({mean_pitch:.0f} Hz)")

            if pitch_std > 40:
                pitch_tags += ["生动", "有感情", "抑扬顿挫"]
            elif pitch_std > 20:
                pitch_tags += ["自然起伏"]
            else:
                pitch_tags += ["平稳", "冷静"]
            lines.append(f"   音高变化: {'丰富' if pitch_std > 40 else '适中' if pitch_std > 20 else '平缓'}")
        else:
            pitch_tags = ["自然"]

        # 音量描述
        if vol_std / max(mean_vol, 0.001) > 1.5:
            vol_tags = ["有爆发力", "情感充沛", "戏剧化"]
        elif vol_std / max(mean_vol, 0.001) > 0.8:
            vol_tags = ["有起伏", "自然"]
        else:
            vol_tags = ["平稳", "均匀"]

        lines.append(f"📢 音量变化: {vol_tags[0]}")

        # 语速描述
        # 粗略：单位时间语音密度
        speech_density = speech_ratio
        if speech_density > 0.7:
            speed_tags = ["语速偏快", "表达密集"]
        elif speech_density > 0.5:
            speed_tags = ["语速适中"]
        else:
            speed_tags = ["语速偏慢", "从容", "缓缓道来"]
        lines.append(f"💬 {speed_tags[0]} (语音占比 {speech_ratio:.0%})")

        # ---- 5. 综合推荐 ----
        all_tags = pitch_tags + vol_tags + speed_tags
        # 去重
        seen = set()
        unique_tags = []
        for t in all_tags:
            if t not in seen:
                seen.add(t)
                unique_tags.append(t)

        lines.append("")
        lines.append("🎯 推荐性格/风格词语")
        lines.append("-" * 40)
        lines.append("、".join(unique_tags))

        # 生成一段可直接用于 instruct 的描述
        instruct_samples = []
        if "沉稳" in unique_tags or "低沉" in unique_tags:
            instruct_samples.append("用沉稳低沉的声音朗读")
        if "生动" in unique_tags or "有感情" in unique_tags:
            instruct_samples.append("用富有感情的语气朗读，注意抑扬顿挫")
        if "语速偏慢" in str(speed_tags):
            instruct_samples.append("放慢语速，从容朗读")
        if "温柔" in str(unique_tags):
            instruct_samples.append("用轻柔温和的语气朗读")
        if "明亮" in unique_tags:
            instruct_samples.append("用明亮轻快的语气朗读")

        if instruct_samples:
            lines.append("")
            lines.append("💡 建议 instruct 指令")
            lines.append("-" * 40)
            for s in instruct_samples:
                lines.append(f"  · {s}")

        lines.append("")
        lines.append(f"⏱️ 音频时长: {duration:.1f}s, 采样率: {sr} Hz")

        return "\n".join(lines)

    except Exception as e:
        return f"❌ 分析失败: {e}"


def preview_segments(text: str, max_seg_len: int) -> str:
    if not text or not text.strip():
        return "⚠️ 请输入文本"
    segs = segment_text(text.strip(), max_chars=max_seg_len)
    lines = [f"📄 总字数: {len(text.strip())} | 分段数: {len(segs)}"]
    lines.append("=" * 50)
    for i, seg in enumerate(segs):
        preview = seg[:80].replace("\n", "\\n")
        if len(seg) > 80:
            preview += "..."
        lines.append(f"段[{i+1:03d}] ({len(seg):4d}字): {preview}")
    lines.append("=" * 50)
    return "\n".join(lines)


def generate_speech(
    text: str,
    language: str,
    max_seg_len: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: int,
    repetition_penalty: float,
    subtalker_temperature: float,
    subtalker_top_k: int,
    subtalker_top_p: float,
    speed: float,
    segment_gap: float,
    breathing_pause: float,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[Optional[str], str, str, str]:
    """Returns: (audio_tuple, clone_status, gen_status, log)"""
    global state

    if not state.model_loaded or state.tts is None:
        return None, "", "❌ 请先加载模型", ""
    if state.prompt_items is None:
        return None, "", "❌ 请先创建音色克隆", ""
    text = text.strip()
    if not text:
        return None, "", "❌ 请输入文本", ""

    segs = segment_text(text, max_chars=max_seg_len)
    if not segs:
        return None, "", "❌ 分段结果为空", ""

    log_lines = [f"📄 文本 {len(text)} 字 → {len(segs)} 段"]
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens, do_sample=True,
        top_k=top_k, top_p=top_p, temperature=temperature,
        repetition_penalty=repetition_penalty,
        subtalker_dosample=True, subtalker_top_k=subtalker_top_k,
        subtalker_top_p=subtalker_top_p, subtalker_temperature=subtalker_temperature,
    )

    # ---- 生成 ----
    all_wavs: List[np.ndarray] = []
    seg_texts: List[str] = []
    sr = state.sample_rate
    total_gen_time = 0.0

    for i, seg_text in enumerate(segs):
        progress((i + 0.5) / len(segs), desc=f"合成段 [{i+1}/{len(segs)}]")
        t0 = time.time()
        try:
            wavs, sr_out = state.tts.generate_voice_clone(
                text=seg_text, language=language,
                voice_clone_prompt=state.prompt_items, **gen_kwargs,
            )
            elapsed = time.time() - t0
            total_gen_time += elapsed
            sr = sr_out
            all_wavs.append(wavs[0])
            seg_texts.append(seg_text)
            log_lines.append(f"  ✅ 段[{i+1}/{len(segs)}] {len(wavs[0])/sr:.1f}s ({elapsed:.1f}s)")
        except Exception as e:
            log_lines.append(f"  ❌ 段[{i+1}] 失败: {e}")

    if not all_wavs:
        return None, "", "❌ 所有段落均生成失败", "\n".join(log_lines)

    # ---- 后处理 ----
    progress(0.92, desc="后处理中...")
    log_lines.append(f"\n🔊 后处理: 语速={speed}x, 段间停顿={segment_gap}s, 气口={breathing_pause}s")

    # 1) 句间气口
    if breathing_pause > 0:
        pwavs = []
        for wav, stxt in zip(all_wavs, seg_texts):
            pwavs.append((insert_breathing_pauses(wav, stxt, sr, breathing_pause), sr))
    else:
        pwavs = [(w, sr) for w in all_wavs]

    # 2) 合并 + 段间停顿
    merged_wav, sr = merge_audio_segments(pwavs, segment_gap=segment_gap)

    # 3) 变速
    if abs(speed - 1.0) > 0.01:
        merged_wav = adjust_speed(merged_wav, sr, speed)

    total_dur = len(merged_wav) / sr
    log_lines.append(f"\n{'='*40}")
    log_lines.append(f"✅ 合成完成! {total_dur:.1f}s ({total_dur/60:.1f} 分钟)")
    log_lines.append(f"   总段数: {len(all_wavs)}, 生成耗时: {total_gen_time:.1f}s")
    log_lines.append(f"{'='*40}")

    return (sr, merged_wav), "✅ 合成完成", f"✅ 完成！{total_dur:.1f}s", "\n".join(log_lines)


# =====================================================================
# Gradio 界面
# =====================================================================

CSS = """
.gradio-container { max-width: 1200px !important; }
.status-box { min-height: 60px; }
.seg-preview { font-family: monospace; font-size: 13px; }
footer { display: none !important; }
"""


def build_ui():
    with gr.Blocks(title="Qwen3-TTS 有声书生成器") as demo:
        gr.Markdown(
            """
            # 🎧 Qwen3-TTS 有声书生成器
            基于阿里 Qwen3-TTS 的开源语音合成工具 · 支持音色克隆 · 长文本分段合成
            """
        )

        # ---- 状态显示 ----
        with gr.Row():
            model_status = gr.Textbox(label="模型状态", value="⏳ 等待加载", elem_classes="status-box")
            clone_status = gr.Textbox(label="音色状态", value="⏳ 等待创建", elem_classes="status-box")
            gen_status = gr.Textbox(label="生成状态", value="⏳ 等待生成", elem_classes="status-box")

        # ---- 第一步：模型设置 ----
        with gr.Accordion("⚙️ 模型设置", open=False):
            with gr.Row():
                model_path = gr.Textbox(
                    label="模型名称/路径", value="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    info="HuggingFace ID 或本地路径（默认 1.7B 最佳音质）", scale=3,
                )
                device = gr.Dropdown(label="设备", choices=["mps", "cpu", "cuda:0"], value="mps", scale=1)
                dtype = gr.Dropdown(label="精度", choices=["float32", "float16", "bfloat16"], value="float16", scale=1)
            with gr.Row():
                load_model_btn = gr.Button("🚀 加载模型", variant="primary")
                unload_model_btn = gr.Button("⏏️ 卸载模型", variant="secondary", size="sm")

        # ---- 第二步：音色克隆 ----
        with gr.Group():
            gr.Markdown("### 🎤 音色克隆")
            with gr.Row():
                ref_audio = gr.Audio(label="参考音频（录制或上传，3 秒以上效果更佳）", type="filepath")
                ref_text = gr.Textbox(
                    label="参考音频文字内容（ICL 模式必填）", lines=3,
                    placeholder="输入参考音频中说话的内容...",
                )
            with gr.Row():
                use_xvec = gr.Checkbox(label="x-vector only 模式（不需要参考文本）", value=False)
                analyze_btn = gr.Button("🔍 分析声音特征", variant="secondary", size="sm")
                clone_btn = gr.Button("🎯 创建音色", variant="primary")
            voice_profile = gr.Textbox(label="声音特征分析", lines=6, max_lines=20, interactive=False, placeholder="点击「分析声音特征」查看结果...")

        # ---- 第三步：文本输入 ----
        with gr.Group():
            gr.Markdown("### 📝 待合成文本")
            with gr.Row():
                text_input = gr.Textbox(label="输入文本", lines=10, placeholder="粘贴或输入要合成有声书的长文本...", scale=3)
                text_file = gr.File(label="或上传 .txt 文件", file_types=[".txt"], scale=1)

        # ---- 参数 ----
        with gr.Accordion("🔧 生成参数", open=False):
            with gr.Row():
                language = gr.Dropdown(
                    label="语言",
                    choices=["Auto", "Chinese", "English", "Japanese", "Korean",
                             "Spanish", "German", "French", "Russian", "Portuguese", "Italian"],
                    value="Auto",
                )
                max_seg_len = gr.Slider(
                    label="每段最大字符数", minimum=50, maximum=2000, value=350, step=50,
                    info="中文建议 300-500，英文建议 500-1000",
                )
                max_new_tokens = gr.Slider(
                    label="每段最大 Token", minimum=256, maximum=8192, value=2048, step=256,
                )
            with gr.Row():
                temperature = gr.Slider(label="温度", minimum=0.1, maximum=2.0, value=0.9, step=0.1)
                top_k = gr.Slider(label="Top-K", minimum=1, maximum=100, value=50, step=1)
                top_p = gr.Slider(label="Top-P", minimum=0.1, maximum=1.0, value=1.0, step=0.05)
                repetition_penalty = gr.Slider(label="重复惩罚", minimum=1.0, maximum=1.5, value=1.05, step=0.01, info="提高可减少重复，增加韵律变化")
            with gr.Row():
                gr.Markdown("##### 🎵 Subtalker（韵律控制）")
            with gr.Row():
                subtalker_temperature = gr.Slider(label="韵律温度", minimum=0.1, maximum=2.0, value=0.9, step=0.1, info="调低=节奏更平更快；调高=节奏变化更多")
                subtalker_top_k = gr.Slider(label="韵律 Top-K", minimum=1, maximum=100, value=50, step=1)
                subtalker_top_p = gr.Slider(label="韵律 Top-P", minimum=0.1, maximum=1.0, value=1.0, step=0.05)

        # ---- 音频后处理参数 ----
        with gr.Accordion("🎛️ 音频后处理（语速/停顿/气口）", open=True):
            with gr.Row():
                speed = gr.Slider(
                    label="语速", minimum=0.5, maximum=1.5, value=0.9, step=0.05,
                    info="0.5=慢一倍, 0.9=稍慢更自然, 1.0=原速",
                )
                segment_gap = gr.Slider(
                    label="段间停顿（秒）", minimum=0.0, maximum=5.0, value=1.5, step=0.1,
                    info="段落之间的静音间隔",
                )
                breathing_pause = gr.Slider(
                    label="气口停顿（秒）", minimum=0.0, maximum=1.0, value=0.25, step=0.05,
                    info="句与句之间的短停顿，0 关闭",
                )

        # ---- 操作按钮 ----
        with gr.Row():
            preview_btn = gr.Button("🔍 预览分段", variant="secondary", size="sm")
            generate_btn = gr.Button("🎬 开始生成有声书", variant="primary", size="lg", scale=2)

        # ---- 分段预览 ----
        seg_preview = gr.Textbox(
            label="分段预览", value="点击「预览分段」查看文本分割结果",
            lines=12, max_lines=25, interactive=False, elem_classes="seg-preview",
        )

        # ---- 结果 ----
        with gr.Row():
            audio_output = gr.Audio(label="生成结果", type="numpy", interactive=False)
        gen_log = gr.Textbox(label="生成日志", lines=8, max_lines=20, interactive=False)

        # ---- 回调 ----
        load_model_btn.click(fn=load_model, inputs=[model_path, device, dtype], outputs=[model_status])
        unload_model_btn.click(fn=unload_model, outputs=[model_status])

        text_file.upload(
            fn=lambda f: Path(f.name).read_text("utf-8") if f else "",
            inputs=[text_file], outputs=[text_input],
        )

        preview_btn.click(fn=preview_segments, inputs=[text_input, max_seg_len], outputs=[seg_preview])

        analyze_btn.click(fn=analyze_voice_profile, inputs=[ref_audio], outputs=[voice_profile])
        clone_btn.click(fn=create_voice_clone, inputs=[ref_audio, ref_text, use_xvec], outputs=[clone_status])

        # 点击生成：先预览分段，再逐段合成
        generate_btn.click(
            fn=preview_segments, inputs=[text_input, max_seg_len], outputs=[seg_preview],
        ).then(
            fn=generate_speech,
            inputs=[text_input, language, max_seg_len, max_new_tokens,
                    temperature, top_k, top_p, repetition_penalty,
                    subtalker_temperature, subtalker_top_k, subtalker_top_p,
                    speed, segment_gap, breathing_pause],
            outputs=[audio_output, clone_status, gen_status, gen_log],
        )

    return demo


# =====================================================================
# 启动
# =====================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-TTS 有声书生成器 WebUI")
    parser.add_argument("--ip", default="127.0.0.1", help="监听地址 (默认: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="监听端口 (默认: 7860)")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    args = parser.parse_args()

    demo = build_ui()
    print(f"\n🌐 启动 WebUI: http://{args.ip}:{args.port}")
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(
            font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
        ),
        css=CSS,
    )
