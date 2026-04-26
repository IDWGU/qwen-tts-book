#!/usr/bin/env python3
"""
Qwen3-TTS 有声书生成器 - WebUI (Gradio)
==========================================

启动:
  source venv/bin/activate
  python webui.py
  # 浏览器打开 http://127.0.0.1:7860
"""

import datetime
import gc
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch

from long_tts import Qwen3TTSModel, segment_text, merge_audio_segments
from long_tts import adjust_speed, insert_breathing_pauses, normalize_audio, split_sentences
from long_tts import init_asr, verify_segment_head_tail

import ref_history

# 模块级 dtype 映射（避免每次 load_model 重新构建 dict）
_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# =====================================================================
# 全局状态（线程安全）
# =====================================================================

class AppState:
    def __init__(self):
        self._lock = threading.Lock()
        self.tts: Optional[Qwen3TTSModel] = None
        self.prompt_items = None
        self.model_loaded = False
        self.model_path = ""
        self.sample_rate = 24000
        # 分段生成工作流
        self.segments: List[str] = []           # 各段文本
        self.segment_audios: List[Optional[np.ndarray]] = []  # 各段音频
        self.segment_sr: int = 24000             # 采样率
        self.source_text: str = ""               # 原始输入文本
        self.session_name: Optional[str] = None  # 当前会话时间戳，如 "20260426195412"
        # 控制台追踪
        self.start_time: float = 0.0             # 会话开始时间戳
        self.segment_times: List[Optional[float]] = []  # 各段生成耗时(s)

    def lock(self):
        return self._lock

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *args):
        self._lock.release()


state = AppState()

# =====================================================================
# 会话管理（持久化到 outputs/<时间戳>/ 目录）
# =====================================================================

PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _list_sessions() -> List[str]:
    """列出所有会话，按时间戳降序（最新的在前）。"""
    if not OUTPUTS_DIR.exists():
        return []
    return sorted(
        (d.name for d in OUTPUTS_DIR.iterdir()
         if d.is_dir() and d.name.isdigit() and len(d.name) >= 12),
        reverse=True,
    )


def _now_ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def _session_dir(timestamp: str) -> Path:
    return OUTPUTS_DIR / timestamp


def _save_session(session_dir: Path, segments: List[str], source_text: str,
                  segment_audios: List[Optional[np.ndarray]], sr: int) -> str:
    """保存音频 + session.json 到会话目录。"""
    session_dir.mkdir(parents=True, exist_ok=True)
    audio_files: Dict[str, str] = {}
    for i, wav in enumerate(segment_audios):
        if wav is not None:
            seg_name = f"seg_{i:04d}.wav"
            sf.write(str(session_dir / seg_name), wav, sr)
            audio_files[str(i)] = seg_name
    meta = {
        "timestamp": session_dir.name,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_text": source_text,
        "segments": segments,
        "audio_files": audio_files,
        "sample_rate": sr,
    }
    (session_dir / "session.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
    )
    return session_dir.name


def _load_session_meta(timestamp: str) -> Optional[dict]:
    p = _session_dir(timestamp) / "session.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _load_session_into_state(timestamp: str) -> Tuple[Optional[dict], int, List[Optional[np.ndarray]]]:
    """加载会话的 meta + 音频到内存。"""
    meta = _load_session_meta(timestamp)
    if meta is None:
        return None, 24000, []
    sr = meta.get("sample_rate", 24000)
    sd = _session_dir(timestamp)
    audio_files = meta.get("audio_files", {})
    audios: List[Optional[np.ndarray]] = [None] * len(meta["segments"])
    for idx_str, fname in audio_files.items():
        idx = int(idx_str)
        wav_path = sd / fname
        if wav_path.exists():
            wav, loaded_sr = sf.read(str(wav_path))
            sr = loaded_sr
            audios[idx] = wav
    return meta, sr, audios


def get_session_label() -> str:
    if state.session_name:
        cnt = sum(1 for a in state.segment_audios if a is not None)
        return f"📁 当前会话: {state.session_name} ({cnt}/{len(state.segments)} 段)"
    return "📁 当前会话: (无)"


def _save_current_session():
    """如果已有分段数据，自动保存到会话目录（首次自动创建）。"""
    if not state.segments:
        return
    if not state.session_name:
        state.session_name = _now_ts()
    sd = _session_dir(state.session_name)
    _save_session(sd, state.segments, state.source_text,
                  state.segment_audios, state.segment_sr)


# =====================================================================
# 控制台（右侧面板）
# =====================================================================

def _build_console_html() -> str:
    """生成右侧控制台面板的 HTML 内容。"""
    with state:
        model_ok = state.model_loaded
        voice_ok = state.prompt_items is not None
        has_text = bool(state.source_text.strip()) if state.source_text else False
        has_segments = len(state.segments) > 0
        seg_count = len(state.segments)
        gen_count = sum(1 for a in state.segment_audios if a is not None)
        seg_times = [t for t in state.segment_times if t is not None]
        start = state.start_time
        merged = False  # will be set when merge is done; can be inferred later

    # 已用时间
    elapsed = time.time() - start if start > 0 else 0
    elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

    # ---- 确定当前工作流阶段 ----
    if not model_ok:
        current_step, step_desc = 1, "加载模型"
    elif not voice_ok:
        current_step, step_desc = 2, "创建音色"
    elif not has_text:
        current_step, step_desc = 3, "输入文本"
    elif not has_segments:
        current_step, step_desc = 4, "预览分段"
    elif gen_count < seg_count:
        current_step, step_desc = 5, "逐段生成"
    else:
        current_step, step_desc = 6, "后处理合并"

    step_names = [
        "① 加载模型",
        "② 创建音色",
        "③ 输入文本",
        "④ 预览分段",
        "⑤ 逐段生成",
        "⑥ 后处理合并",
    ]

    steps_html = ""
    for i, name in enumerate(step_names):
        idx = i + 1
        if idx < current_step:
            icon, cls = "✅", "step-done"
        elif idx == current_step:
            icon, cls = "▶", "step-active"
        else:
            icon, cls = "⏳", "step-pending"
        if idx == 5 and seg_count > 0:
            label = f"{name}  [{gen_count}/{seg_count} 段]"
        else:
            label = name
        steps_html += f'<div class="console-step {cls}">{icon} {label}</div>'

    # ---- 统计信息 ----
    stats_html = ""
    stats_html += f'<div class="console-stat"><span class="stat-label">已用时间</span><span class="stat-value">{elapsed_str}</span></div>'
    stats_html += f'<div class="console-stat"><span class="stat-label">总字数</span><span class="stat-value">{len(state.source_text):,}</span></div>'
    stats_html += f'<div class="console-stat"><span class="stat-label">总段数</span><span class="stat-value">{seg_count}</span></div>'
    stats_html += f'<div class="console-stat"><span class="stat-label">已生成</span><span class="stat-value">{gen_count}/{seg_count}</span></div>'
    stats_html += f'<div class="console-stat"><span class="stat-label">会话</span><span class="stat-value">{state.session_name or "—"}</span></div>'
    if merged:
        stats_html += '<div class="console-stat"><span class="stat-label">合并</span><span class="stat-value">✅ 已完成</span></div>'

    # ---- 各段用时 ----
    timing_html = ""
    if seg_times:
        # 只显示最近 20 段，避免太长
        display_times = seg_times[-20:]
        for i, t in enumerate(display_times):
            real_idx = len(seg_times) - len(display_times) + i
            timing_html += f'<div class="console-timing">段 {real_idx+1}: {t:.1f}s</div>'
        if len(seg_times) > 20:
            timing_html += f'<div class="console-timing">...（共 {len(seg_times)} 段）</div>'

    html = f"""<div class="console-panel">
  <h3 class="console-title">🎛 控制台</h3>

  <div class="console-section">
    <div class="console-section-title">▶ 工作流</div>
    {steps_html}
    <div class="console-hint">当前阶段: {step_desc}</div>
  </div>

  <div class="console-section">
    <div class="console-section-title">📊 统计</div>
    {stats_html}
  </div>

  <div class="console-section">
    <div class="console-section-title">⏱ 各段用时</div>
    {timing_html if timing_html else '<div class="console-timing" style="color:#888">暂无数据</div>'}
  </div>
</div>"""
    return html


def _resolve_local_cache(repo_id: str) -> Optional[str]:
    """Resolve a HuggingFace repo ID to a local cache snapshot path if cached.

    按优先级检查 refs：main → master → 第一个存在的 ref 文件。
    """
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir_name = "models--" + repo_id.replace("/", "--")
    repo_dir = cache_dir / repo_dir_name
    if not repo_dir.exists():
        return None
    refs_dir = repo_dir / "refs"
    if not refs_dir.is_dir():
        return None

    # 按优先级尝试不同的 ref 名称
    for ref_name in ("main", "master"):
        refs_file = refs_dir / ref_name
        if refs_file.is_file():
            commit_hash = refs_file.read_text().strip()
            snapshot_dir = repo_dir / "snapshots" / commit_hash
            if snapshot_dir.is_dir():
                return str(snapshot_dir)

    # 兜底：使用 refs 目录下的第一个文件
    try:
        first_ref = next(refs_dir.iterdir())
        if first_ref.is_file():
            commit_hash = first_ref.read_text().strip()
            snapshot_dir = repo_dir / "snapshots" / commit_hash
            if snapshot_dir.is_dir():
                return str(snapshot_dir)
    except (StopIteration, OSError):
        pass

    return None


# =====================================================================
# 核心逻辑
# =====================================================================

def load_model(model_path: str, device: str, dtype_str: str) -> Tuple[str, str]:
    with state:
        if state.model_loaded and state.model_path == model_path:
            return f"✅ 模型已加载: {model_path}", _build_console_html()

    # 自动解析 repo ID 到本地缓存路径，避免 huggingface_hub 网络重试
    resolved = _resolve_local_cache(model_path) or model_path
    if resolved != model_path:
        print(f"[load_model] 自动解析到本地缓存: {resolved}")

    dtype = _DTYPE_MAP.get(dtype_str, torch.float32)
    if device == "mps" and dtype == torch.bfloat16:
        dtype = torch.float32
    attn_impl = "flash_attention_2" if device.startswith("cuda") else None

    try:
        t0 = time.time()
        tts = Qwen3TTSModel.from_pretrained(
            resolved,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
            local_files_only=True,
        )

        with state:
            state.tts = tts
            state.model_loaded = True
            state.model_path = model_path
            state.prompt_items = None  # 模型变更后重置音色
        return f"✅ 模型加载完成 ({time.time() - t0:.1f}s)", _build_console_html()
    except Exception as e:
        with state:
            state.model_loaded = False
            state.tts = None
        return f"❌ 模型加载失败: {e}", _build_console_html()


def unload_model() -> Tuple[str, str]:
    with state:
        tts = state.tts
        if tts is None:
            return "⏳ 模型未加载", _build_console_html()
        state.tts = None
        state.prompt_items = None
        state.model_loaded = False
        state.model_path = ""
    try:
        del tts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # MPS （macOS GPU）也需要清理缓存
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        return "✅ 模型已从内存卸载", _build_console_html()
    except Exception as e:
        return f"❌ 卸载失败: {e}", _build_console_html()


def load_from_history(selected: str) -> Tuple[gr.update, gr.update, gr.update]:
    """从历史记录加载音频和文本到输入框。"""
    result = ref_history.resolve(selected)
    if result is None:
        return (
            gr.update(value=None),
            gr.update(value=""),
            gr.update(value=False),
        )
    audio_path, text, xvec = result
    return (
        gr.update(value=audio_path),
        gr.update(value=text),
        gr.update(value=xvec),
    )


def create_voice_clone(
    audio_path: str, ref_text: str, use_xvec: bool, ref_name: str,
) -> Tuple[str, gr.update, str]:
    """创建音色克隆，同时自动保存到历史记录。"""
    with state as s:
        if not s.model_loaded or s.tts is None:
            return "❌ 请先加载模型", gr.update(), _build_console_html()
        tts = s.tts
    if not audio_path:
        return "❌ 请上传参考音频", gr.update(), _build_console_html()
    try:
        if use_xvec:
            prompt_items = tts.create_voice_clone_prompt(
                ref_audio=audio_path, x_vector_only_mode=True,
            )
        else:
            if not ref_text or not ref_text.strip():
                return "❌ ICL 模式需要提供参考音频的文字内容", gr.update(), _build_console_html()
            prompt_items = tts.create_voice_clone_prompt(
                ref_audio=audio_path, ref_text=ref_text.strip(), x_vector_only_mode=False,
            )
        with state:
            state.prompt_items = prompt_items

        # 自动保存到历史记录
        if ref_name and ref_name.strip():
            ref_history.add(
                name=ref_name.strip(),
                audio_src=audio_path,
                text=ref_text.strip() if ref_text else "",
                xvec=use_xvec,
            )

        # 更新下拉框选项
        new_choices = ["-- 新建参考音频 --"] + ref_history.list_names()
        msg = "✅ 音色克隆完成"
        if ref_name and ref_name.strip():
            msg += f"，已保存为「{ref_name.strip()}」"
        return msg, gr.update(choices=new_choices), _build_console_html()

    except Exception as e:
        with state:
            state.prompt_items = None
        return f"❌ 音色克隆失败: {e}", gr.update(), _build_console_html()


def _vad_speech_ratio(y: np.ndarray, sr: int,
                      rms: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """基于能量阈值的简单 VAD，带边缘保护。

    接收可选的预计算 rms 数组以避免重复计算。

    返回: (语音比例, 时长秒数)
    """
    if len(y) < sr * 0.05:  # 小于 50ms 视为静音
        return 0.0, len(y) / max(sr, 1)
    if rms is None:
        rms = librosa.feature.rms(y=y)[0]
    energy = float(np.mean(rms))
    if energy < 1e-10:  # 完全静音
        return 0.0, len(y) / sr
    threshold = energy * 0.3
    is_speech = rms > threshold
    speech_ratio = float(np.sum(is_speech)) / max(len(rms), 1)
    return speech_ratio, len(y) / sr


def _pitch_analysis(y: np.ndarray, sr: int,
                    rms: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """提取音高特征，带空音频保护。

    接收可选的预计算 rms 数组以避免重复计算。

    返回: (平均音高Hz, 音高标准差)
    """
    if len(y) < sr * 0.1:  # 小于 100ms 无法分析
        return 0.0, 0.0
    if rms is None:
        rms = librosa.feature.rms(y=y)[0]
    if float(np.mean(rms)) < 1e-10:  # 静音
        return 0.0, 0.0
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=65, fmax=2093, sr=sr)
        pitched = f0[voiced_flag] if voiced_flag.any() else np.array([])
        if len(pitched) == 0:
            return 0.0, 0.0
        return float(np.mean(pitched)), float(np.std(pitched))
    except Exception:
        return 0.0, 0.0


def analyze_voice_profile(audio_path: str) -> str:
    """分析参考音频，提取声音特征并推荐性格描述词语。"""
    if not audio_path:
        return "⚠️ 请先上传参考音频"

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        if len(y) < sr * 0.5:  # < 0.5s
            return "⚠️ 音频太短（< 0.5 秒），请使用更长的参考音频（建议 3 秒以上）"

        # ---- 统一计算 RMS（仅一次，避免重复 librosa.feature.rms 调用） ----
        rms = librosa.feature.rms(y=y)[0]
        if float(np.mean(rms)) < 1e-8:
            return "⚠️ 音频似乎静音，请上传包含人声的音频文件"

        # ---- 1. 音高分析（传入预计算 rms） ----
        mean_pitch, pitch_std = _pitch_analysis(y, sr, rms)

        # ---- 2. 能量/音量 ----
        mean_vol = float(np.mean(rms))
        vol_std = float(np.std(rms))

        # ---- 3. 语速（传入预计算 rms） ----
        speech_ratio, duration = _vad_speech_ratio(y, sr, rms)

        # ---- 4. 生成声音画像 ----
        lines = []
        lines.append("📊 声音特征分析")
        lines.append("=" * 40)

        # 音高描述
        if mean_pitch > 0:
            if mean_pitch > 220:
                pitch_desc = "较高"
                pitch_tags: List[str] = ["明亮", "清亮"]
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

        # 音量描述（除零保护）
        vol_ratio = vol_std / max(mean_vol, 1e-10)
        if vol_ratio > 1.5:
            vol_tags = ["有爆发力", "情感充沛", "戏剧化"]
        elif vol_ratio > 0.8:
            vol_tags = ["有起伏", "自然"]
        else:
            vol_tags = ["平稳", "均匀"]

        lines.append(f"📢 音量变化: {vol_tags[0]}")

        # 语速描述
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
        # 去重（保持顺序）
        seen = set()
        unique_tags = [t for t in all_tags if not (t in seen or seen.add(t))]

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


def get_segment_status_display() -> str:
    """Build status text showing each segment's state."""
    with state:
        if not state.segments:
            return "⏳ 请先分段文本"
        lines = [f"📄 共 {len(state.segments)} 段"]
        for i, seg in enumerate(state.segments):
            audio = state.segment_audios[i] if i < len(state.segment_audios) else None
            icon = "✅" if audio is not None else "⏳"
            preview = seg[:40].replace("\n", "\\n")
            if len(seg) > 40:
                preview += "..."
            lines.append(f"  {icon} 段 [{i+1}] ({len(seg)}字): {preview}")
    return "\n".join(lines)


def do_segment(text: str, max_seg_len: int) -> Tuple[str, gr.update, str, str]:
    """Segment text, store in state, return preview + dropdown update + status."""
    if not text or not text.strip():
        with state:
            state.segments = []
            state.segment_audios = []
            state.segment_times = []
            state.source_text = ""
        return "⚠️ 请输入文本", gr.update(choices=[], value=None, interactive=False), "", _build_console_html()

    segs = segment_text(text.strip(), max_chars=max_seg_len)
    with state:
        state.segments = segs
        state.segment_audios = [None] * len(segs)
        state.segment_times = [None] * len(segs)
        state.segment_sr = 24000
        state.source_text = text.strip()
        # 计时开始
        if state.start_time == 0:
            state.start_time = time.time()

    preview = preview_segments(text, max_seg_len)
    choices = [f"段 {i+1} ({len(seg)}字)" for i, seg in enumerate(segs)]
    status = get_segment_status_display()

    return preview, gr.update(
        choices=choices, value=choices[0] if choices else None, interactive=True
    ), status, _build_console_html()


def generate_segment(
    seg_label: str, language: str, max_new_tokens: int,
    temperature: float, top_k: int, top_p: int, repetition_penalty: float,
    subtalker_temperature: float, subtalker_top_k: int, subtalker_top_p: float,
    enable_asr: bool = False,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str, str, str]:
    """Generate audio for a single selected segment."""
    with state as s:
        if not s.model_loaded or s.tts is None:
            return None, "❌ 请先加载模型", get_session_label(), _build_console_html()
        if s.prompt_items is None:
            return None, "❌ 请先创建音色克隆", get_session_label(), _build_console_html()
        if not s.segments:
            return None, "❌ 请先分段文本", get_session_label(), _build_console_html()
        tts = s.tts
        prompt_items = s.prompt_items
        segments = list(s.segments)
    if not seg_label:
        return None, "❌ 请选择段落", get_session_label(), _build_console_html()

    try:
        seg_idx = int(seg_label.split()[1]) - 1
    except (ValueError, IndexError):
        return None, "❌ 无效的段落选择", get_session_label(), _build_console_html()
    if seg_idx < 0 or seg_idx >= len(segments):
        return None, "❌ 段号超出范围", get_session_label(), _build_console_html()

    seg_text = segments[seg_idx]

    # 计时开始
    t0 = time.time()

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens, do_sample=True,
        top_k=top_k, top_p=top_p, temperature=temperature,
        repetition_penalty=repetition_penalty,
        subtalker_dosample=True, subtalker_top_k=subtalker_top_k,
        subtalker_top_p=subtalker_top_p, subtalker_temperature=subtalker_temperature,
    )

    # ASR 校验（需要的话延迟初始化）
    asr_available = False
    if enable_asr:
        asr_available = init_asr(quiet=True)

    retry_kwargs = dict(gen_kwargs)
    max_retry = 3 if enable_asr else 0
    best_wav = None
    best_sr = None
    success = False
    asr_result = ""  # 校验结果信息

    for attempt in range(max_retry + 1):
        try:
            wavs, sr = tts.generate_voice_clone(
                text=seg_text, language=language,
                voice_clone_prompt=prompt_items, **retry_kwargs,
            )
            seg_wav = wavs[0]

            if not enable_asr or not asr_available:
                # 无校验 → 直接成功
                elapsed = time.time() - t0
                with state:
                    state.segment_audios[seg_idx] = seg_wav
                    state.segment_sr = sr
                    if seg_idx < len(state.segment_times):
                        state.segment_times[seg_idx] = elapsed
                _save_current_session()
                console = _build_console_html()
                return (sr, seg_wav), get_segment_status_display(), get_session_label(), console

            # ---- ASR 校验 ----
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            sf.write(tmp_path, seg_wav, sr)

            try:
                passed, asr_text, asr_conf, fail_reason = verify_segment_head_tail(
                    tmp_path, seg_text, language=language,
                )
            finally:
                os.unlink(tmp_path)

            if passed:
                elapsed = time.time() - t0
                with state:
                    state.segment_audios[seg_idx] = seg_wav
                    state.segment_sr = sr
                    if seg_idx < len(state.segment_times):
                        state.segment_times[seg_idx] = elapsed
                _save_current_session()
                asr_result = f" ✅ ASR✓"
                status = f"✅ 段 [{seg_idx+1}] 生成完成{asr_result}\n{get_segment_status_display()}"
                console = _build_console_html()
                return (sr, seg_wav), status, get_session_label(), console
            else:
                if attempt == 0:
                    best_wav, best_sr = seg_wav, sr
                asr_result = f" ⚠️ ASR: {fail_reason}"
                if attempt < max_retry:
                    retry_kwargs["temperature"] = min(retry_kwargs["temperature"] + 0.1, 1.2)
                else:
                    # 最后一次也失败了，保留最佳版本
                    elapsed = time.time() - t0
                    with state:
                        state.segment_audios[seg_idx] = best_wav
                        state.segment_sr = best_sr or sr
                        if seg_idx < len(state.segment_times):
                            state.segment_times[seg_idx] = elapsed
                    _save_current_session()
                    status = f"⚠️ 段 [{seg_idx+1}] ASR 校验失败，已保留{asr_result}\n{get_segment_status_display()}"
                    console = _build_console_html()
                    return (best_sr or sr, best_wav), status, get_session_label(), console

        except Exception as e:
            return None, f"❌ 段 [{seg_idx+1}] 生成失败: {e}\n\n{get_segment_status_display()}", get_session_label(), _build_console_html()

    return None, "❌ 未知错误", get_session_label(), _build_console_html()


def generate_all_segments(
    language: str, max_new_tokens: int,
    temperature: float, top_k: int, top_p: int, repetition_penalty: float,
    subtalker_temperature: float, subtalker_top_k: int, subtalker_top_p: float,
    enable_asr: bool = False,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[Optional[Tuple[int, np.ndarray]], str, str, str]:
    """Generate all un-generated segments sequentially with optional ASR verification.

    优化：单次加锁预计算 needed 列表，避免每段生成时重复 lock acquire/release。
    """
    with state as s:
        if not s.model_loaded or s.tts is None:
            return None, "❌ 请先加载模型", get_session_label(), _build_console_html()
        if s.prompt_items is None:
            return None, "❌ 请先创建音色克隆", get_session_label(), _build_console_html()
        if not s.segments:
            return None, "❌ 请先分段文本", get_session_label(), _build_console_html()
        tts = s.tts
        prompt_items = s.prompt_items
        segments = list(s.segments)
        total = len(segments)
        # 预计算需要生成的段落索引（单次加锁，避免循环中重复 acquire/release）
        needed = [i for i, a in enumerate(s.segment_audios) if a is None]

    if not needed:
        return None, "✅ 所有段落已生成\n\n💡 请往下滚动到「音频后处理」区域，调整参数后点击「合并并输出」。\n\n" + get_segment_status_display(), get_session_label(), _build_console_html()

    # ASR 校验（延迟初始化）
    asr_available = False
    if enable_asr:
        asr_available = init_asr(quiet=True)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens, do_sample=True,
        top_k=top_k, top_p=top_p, temperature=temperature,
        repetition_penalty=repetition_penalty,
        subtalker_dosample=True, subtalker_top_k=subtalker_top_k,
        subtalker_top_p=subtalker_top_p, subtalker_temperature=subtalker_temperature,
    )

    last_audio = None
    errors = 0
    need_count = len(needed)
    verify_checked = 0
    verify_passed = 0
    verify_retried = 0
    verify_failed = 0

    for seq_idx, i in enumerate(needed):
        progress((seq_idx + 0.5) / need_count, desc=f"合成段 [{i+1}/{total}]")
        seg_text = segments[i]
        t0 = time.time()

        retry_kwargs = dict(gen_kwargs)
        max_retry = 3 if enable_asr and asr_available else 0
        seg_success = False
        best_wav = None
        best_sr_local = None

        for attempt in range(max_retry + 1):
            try:
                wavs, sr_local = tts.generate_voice_clone(
                    text=seg_text, language=language,
                    voice_clone_prompt=prompt_items, **retry_kwargs,
                )
                seg_wav = wavs[0]
                if attempt == 0:
                    best_wav, best_sr_local = seg_wav, sr_local

                if not enable_asr or not asr_available:
                    # 无校验 → 直接保存
                    elapsed = time.time() - t0
                    with state:
                        state.segment_audios[i] = seg_wav
                        state.segment_sr = sr_local
                        if i < len(state.segment_times):
                            state.segment_times[i] = elapsed
                    last_audio = (sr_local, seg_wav)
                    seg_success = True
                    break

                # ---- ASR 校验 ----
                verify_checked += 1
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_path = tmp.name
                tmp.close()
                sf.write(tmp_path, seg_wav, sr_local)

                try:
                    passed, asr_text, asr_conf, fail_reason = verify_segment_head_tail(
                        tmp_path, seg_text, language=language,
                    )
                finally:
                    os.unlink(tmp_path)

                if passed:
                    verify_passed += 1
                    elapsed = time.time() - t0
                    with state:
                        state.segment_audios[i] = seg_wav
                        state.segment_sr = sr_local
                        if i < len(state.segment_times):
                            state.segment_times[i] = elapsed
                    last_audio = (sr_local, seg_wav)
                    seg_success = True
                    break
                else:
                    if attempt == 0:
                        best_wav, best_sr_local = seg_wav, sr_local
                    if attempt < max_retry:
                        verify_retried += 1
                        retry_kwargs["temperature"] = min(
                            retry_kwargs["temperature"] + 0.1, 1.2
                        )
                        progress((seq_idx + 0.5) / need_count,
                                 desc=f"段 [{i+1}] ASR 重试 {attempt+1}/{max_retry}")
                    else:
                        verify_failed += 1
                        elapsed = time.time() - t0
                        with state:
                            state.segment_audios[i] = best_wav
                            state.segment_sr = best_sr_local or sr_local
                            if i < len(state.segment_times):
                                state.segment_times[i] = elapsed
                        last_audio = (best_sr_local or sr_local, best_wav)
                        seg_success = True
                        break

            except Exception:
                if not seg_success:
                    errors += 1
                break

        if not seg_success:
            errors += 1

    # 自动保存到会话
    _save_current_session()

    with state:
        done = sum(1 for a in state.segment_audios if a is not None)

    summary_lines = [f"✅ 已完成 {done}/{total} 段"]
    if errors:
        summary_lines.append(f" ({errors} 段失败)")

    # ASR 校验报告
    if enable_asr and asr_available and verify_checked > 0:
        pct = verify_passed / max(verify_checked, 1) * 100
        summary_lines.append(f"\n📊 ASR 校验：{verify_checked} 段校验")
        summary_lines.append(f"   ✅ 一次通过: {verify_passed} ({pct:.0f}%)")
        if verify_retried:
            summary_lines.append(f"   ↻ 重试修正: {verify_retried}")
        if verify_failed:
            summary_lines.append(f"   ⚠️ 校验失败: {verify_failed}（已保留）")

    if done == total and not errors:
        summary_lines.append("\n💡 所有段落已生成完成！请往下滚动到「音频后处理」区域，调整参数后点击「合并并输出」。")
    elif done == total:
        summary_lines.append("\n所有可用段落已生成完成")
    else:
        summary_lines.append("\n部分段落尚未生成，可继续生成")

    summary = "\n".join(summary_lines)
    return last_audio, f"{summary}\n{get_segment_status_display()}", get_session_label(), _build_console_html()


def merge_segments(
    speed: float, segment_gap: float, breathing_pause: float,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[Optional[Tuple[int, np.ndarray]], str, str, str]:
    """Merge all generated segments into final audio.

    合并后保留各段音频在 state 中，方便调整后处理参数后再次合并。
    """
    with state as s:
        if not s.segment_audios or all(a is None for a in s.segment_audios):
            return None, "❌ 没有已生成的段落，请先生成", get_session_label(), _build_console_html()

        generated = [(i, a) for i, a in enumerate(s.segment_audios) if a is not None]
        total = len(s.segment_audios)
        done = len(generated)
        sr = s.segment_sr
        all_wavs = [a for _, a in generated]
        seg_texts = [s.segments[i] for i, _ in generated]
        # 注意：不释放 segment_audios，保留分段数据供调整参数后再次合并

    warn = ""
    if done < total:
        n_missing = total - done
        if n_missing == total:
            return None, "❌ 所有段落均未生成，请先生成", get_session_label(), _build_console_html()
        warn = f"⚠️ 还有 {n_missing} 段未生成，仅合并已生成的 {done}/{total} 段\n\n"

    progress(0.1, desc="后处理中...")
    pwavs = []
    for wav, stxt in zip(all_wavs, seg_texts):
        # 音频归一化：淡入 + 音量统一 → 消除开头爆音和忽大忽小
        wav = normalize_audio(wav, sr)
        if breathing_pause > 0:
            wav = insert_breathing_pauses(wav, stxt, sr, breathing_pause)
        pwavs.append((wav, sr))

    # 进入后处理前释放中间列表引用，帮助 GC 回收
    del all_wavs, seg_texts

    merged_wav, sr = merge_audio_segments(pwavs, segment_gap=segment_gap)
    del pwavs  # 合并完成，pwavs 中的数组已在 out 中复制完毕

    if abs(speed - 1.0) > 0.01:
        progress(0.5, desc="变速处理中...")
        merged_wav = adjust_speed(merged_wav, sr, speed)

    # 保存合并结果到会话目录
    merged_path = ""
    with state:
        if state.session_name:
            sd = _session_dir(state.session_name)
            sd.mkdir(parents=True, exist_ok=True)
            mf = sd / "merged.wav"
            sf.write(str(mf), merged_wav, sr)
            merged_path = str(mf)
            # 更新 session.json 中的 merged_file
            meta_path = sd / "session.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["merged_file"] = "merged.wav"
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    total_dur = len(merged_wav) / sr
    path_info = f"\n   📁 已保存: {merged_path}" if merged_path else ""
    status = (
        f"{warn}🔗 合并完成! {total_dur:.1f}s ({total_dur/60:.1f} 分钟)"
        f"{path_info}\n"
        f"   不满意可重新生成某段后再点合并\n"
        f"{get_segment_status_display()}"
    )

    return (sr, merged_wav), status, get_session_label(), _build_console_html()


# =====================================================================
# Gradio 界面
# =====================================================================

CSS = """
.gradio-container { max-width: 1400px !important; }
.status-box { min-height: 60px; }
.seg-preview { font-family: monospace; font-size: 13px; }
footer { display: none !important; }

/* ---- 右侧控制台 ---- */
.console-panel {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 16px;
    font-size: 13px;
    line-height: 1.6;
    height: calc(100vh - 120px);
    overflow-y: auto;
    border: 1px solid #e0e0e0;
}
.console-title {
    margin: 0 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #4a90d9;
    font-size: 15px;
    color: #333;
}
.console-section {
    margin-bottom: 14px;
}
.console-section-title {
    font-weight: 600;
    font-size: 13px;
    color: #555;
    margin-bottom: 6px;
}
.console-step {
    padding: 3px 0;
    font-size: 12.5px;
}
.console-step.step-done { color: #2e7d32; }
.console-step.step-active { color: #1565c0; font-weight: 600; }
.console-step.step-pending { color: #999; }
.console-hint {
    margin-top: 4px;
    font-size: 12px;
    color: #888;
    font-style: italic;
}
.console-stat {
    display: flex;
    justify-content: space-between;
    padding: 2px 0;
    font-size: 12.5px;
    border-bottom: 1px solid #eee;
}
.console-stat .stat-label { color: #666; }
.console-stat .stat-value { font-weight: 600; color: #333; }
.console-timing {
    font-size: 12px;
    color: #555;
    padding: 1px 0;
    font-family: monospace;
}

/* ---- 左栏布局统一高度 ---- */
.aligned-row {
    display: flex;
    align-items: stretch;
}
.aligned-row > * {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
}
.btn-equal {
    height: 100% !important;
    min-height: 38px !important;
}
/* Propagate height through internal Gradio wrappers */
.aligned-row .gr-box,
.aligned-row .gr-form-item {
    height: 100%;
    display: flex;
    flex-direction: column;
}
.aligned-row .wrap {
    height: 100%;
    display: flex;
    flex-direction: column;
    min-height: 0;
}
/* Force the actual input control (last child of wrap) to fill remaining space */
.aligned-row .wrap > :not(label) {
    flex: 1;
    min-height: 0;
}
"""


# =====================================================================
# 会话加载 / 新建 / 选择响应
# =====================================================================


def do_new_session() -> Tuple[str, gr.update, str, str, str, Optional[str], str]:
    """清空当前工作区，创建新会话。"""
    global state
    with state:
        state.segments = []
        state.segment_audios = []
        state.segment_times = []
        state.source_text = ""
        state.session_name = _now_ts()
        state.start_time = 0
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = _list_sessions()
    return (
        "📭 已创建新会话，请输入文本并点击「预览分段」",
        gr.update(choices=[], value=None, interactive=False),
        "⏳ 等待分段",
        get_session_label(),
        "",
        None,  # 清空音频预览
        _build_console_html(),
    )


def do_load_session(timestamp: str) -> Tuple[str, gr.update, str, str, str, Optional[str], str]:
    """加载历史会话，恢复所有分段和音频。"""
    global state
    if not timestamp:
        return "⚠️ 请选择一个会话", gr.update(), get_session_label(), "", "", None, _build_console_html()

    meta, sr, audios = _load_session_into_state(timestamp)
    if meta is None:
        return f"❌ 会话 {timestamp} 不存在", gr.update(), get_session_label(), "", "", None, _build_console_html()

    with state:
        state.segments = meta["segments"]
        state.segment_audios = audios
        state.segment_sr = sr
        state.source_text = meta.get("source_text", "")
        state.session_name = timestamp
        state.start_time = time.time()
        state.segment_times = [None] * len(meta["segments"])

    preview = preview_segments(state.source_text or " ".join(state.segments), 500)
    choices = [f"段 {i+1} ({len(seg)}字)" for i, seg in enumerate(state.segments)]
    return (
        preview,
        gr.update(choices=choices, value=choices[0] if choices else None, interactive=True),
        get_segment_status_display(),
        get_session_label(),
        state.source_text,
        None,  # 清空之前的播放器
        _build_console_html(),
    )


def on_segment_select(seg_label: str) -> Optional[Tuple[int, np.ndarray]]:
    """当选择器切换时，如果该段已有音频就显示在试听中。"""
    if not seg_label or not state.segments:
        return None
    try:
        idx = int(seg_label.split()[1]) - 1
    except (ValueError, IndexError):
        return None
    with state:
        if 0 <= idx < len(state.segment_audios) and state.segment_audios[idx] is not None:
            return (state.segment_sr, state.segment_audios[idx])
    return None


def build_ui():
    with gr.Blocks(title="Qwen3-TTS 有声书生成器") as demo:
        gr.Markdown(
            """
            # 🎧 Qwen3-TTS 有声书生成器
            基于阿里 Qwen3-TTS 的开源语音合成工具 · 支持音色克隆 · 长文本分段合成 · 自动保存到 outputs/
            """
        )

        # ---- 状态显示（顶部通栏） ----
        with gr.Row():
            model_status = gr.Textbox(label="模型状态", value="⏳ 等待加载", elem_classes="status-box")
            clone_status = gr.Textbox(label="音色状态", value="⏳ 等待创建", elem_classes="status-box")
            session_label = gr.Textbox(label="会话", value=get_session_label(), elem_classes="status-box")

        # ---- 主体：左（操作区）+ 右（控制台） ----
        with gr.Row():
            # =============================================================
            # 左栏：操作界面
            # =============================================================
            with gr.Column(scale=3):
                # ---- 会话管理 ----
                with gr.Row(elem_classes="aligned-row"):
                    session_selector = gr.Dropdown(
                        label="📂 历史会话", choices=_list_sessions(),
                        interactive=True, scale=3,
                        info="选择之前的合成结果来继续处理",
                    )
                    load_session_btn = gr.Button("📥 加载会话", variant="secondary", scale=1, elem_classes="btn-equal")
                    new_session_btn = gr.Button("📄 新建会话", variant="secondary", scale=1, elem_classes="btn-equal")

                # ---- 第一步：模型设置 ----
                with gr.Accordion("⚙️ 模型设置", open=False):
                    with gr.Row(elem_classes="aligned-row"):
                        model_path = gr.Textbox(
                            label="模型名称/路径", value="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                            info="HuggingFace ID 或本地路径（默认 1.7B 最佳音质）", scale=3,
                        )
                        device = gr.Dropdown(label="设备", choices=["mps", "cpu", "cuda:0"], value="mps", scale=1)
                        dtype = gr.Dropdown(label="精度", choices=["float32", "float16", "bfloat16"], value="float16", scale=1)
                    with gr.Row(elem_classes="aligned-row"):
                        load_model_btn = gr.Button("🚀 加载模型", variant="primary", elem_classes="btn-equal")
                        unload_model_btn = gr.Button("⏏️ 卸载模型", variant="secondary", elem_classes="btn-equal")

                # ---- 第二步：音色克隆 ----
                with gr.Group():
                    gr.Markdown("### 🎤 音色克隆")
                    with gr.Row(elem_classes="aligned-row"):
                        ref_history_dd = gr.Dropdown(
                            label="📂 参考音频历史",
                            choices=["-- 新建参考音频 --"] + ref_history.list_names(),
                            value="-- 新建参考音频 --",
                            interactive=True, scale=2,
                            info="选择已保存的参考音频，点击「加载历史」自动填充",
                        )
                        ref_name = gr.Textbox(
                            label="引用名称", scale=2,
                            placeholder="填写名称，创建音色时将自动保存",
                        )
                        load_history_btn = gr.Button("📂 加载历史", variant="secondary", scale=1, elem_classes="btn-equal")
                    with gr.Row(elem_classes="aligned-row"):
                        ref_audio = gr.Audio(label="参考音频（录制或上传，3 秒以上效果更佳）", type="filepath")
                        ref_text = gr.Textbox(
                            label="参考音频文字内容（ICL 模式必填）", lines=3,
                            placeholder="输入参考音频中说话的内容...",
                        )
                    with gr.Row(elem_classes="aligned-row"):
                        use_xvec = gr.Checkbox(label="x-vector only 模式（不需要参考文本）", value=False)
                        analyze_btn = gr.Button("🔍 分析声音特征", variant="secondary", elem_classes="btn-equal")
                        clone_btn = gr.Button("🎯 创建音色", variant="primary", elem_classes="btn-equal")
                    voice_profile = gr.Textbox(label="声音特征分析", lines=6, max_lines=20, interactive=False, placeholder="点击「分析声音特征」查看结果...")

                # ---- 第三步：文本输入 ----
                with gr.Group():
                    gr.Markdown("### 📝 待合成文本")
                    with gr.Row(elem_classes="aligned-row"):
                        text_input = gr.Textbox(label="输入文本", lines=10, placeholder="粘贴或输入要合成有声书的长文本...", scale=3)
                        text_file = gr.File(label="或上传 .txt 文件", file_types=[".txt"], scale=1)

                # ---- 参数 ----
                with gr.Accordion("🔧 生成参数", open=False):
                    with gr.Row(elem_classes="aligned-row"):
                        language = gr.Dropdown(
                            label="语言",
                            choices=["Auto", "Chinese", "English", "Japanese", "Korean",
                                     "Spanish", "German", "French", "Russian", "Portuguese", "Italian"],
                            value="Auto",
                        )
                        max_seg_len = gr.Slider(
                            label="每段最大字符数", minimum=50, maximum=2000, value=350, step=50,
                            info="决定文本被切分成多少段。值越小段数越多（每段更短），值越大段数越少（每段更长）。中文建议 300-500，英文建议 500-1000",
                        )
                        max_new_tokens = gr.Slider(
                            label="每段最大 Token", minimum=256, maximum=8192, value=2048, step=256,
                            info="控制每段生成的语音长度。数值越大，一次生成的语音越长，但超出此长度会被截断。建议 2048",
                        )
                    with gr.Row(elem_classes="aligned-row"):
                        temperature = gr.Slider(
                            label="温度", minimum=0.1, maximum=2.0, value=0.9, step=0.1,
                            info="控制生成的随机性。调低 (=0.6)= 稳定可预测，适合小说朗读；调高 (>1.2)= 更有创意但可能不稳定",
                        )
                        top_k = gr.Slider(
                            label="Top-K", minimum=1, maximum=100, value=50, step=1,
                            info="限制模型每次只从概率最高的 K 个候选词中选择。调低 (=20)= 更保守；调高 (=80)= 更多变化",
                        )
                        top_p = gr.Slider(
                            label="Top-P", minimum=0.1, maximum=1.0, value=1.0, step=0.05,
                            info="累积概率采样，与 Top-K 配合使用。调低 (=0.8)= 更稳定；调高 (=1.0)= 保留全部可能性",
                        )
                        repetition_penalty = gr.Slider(
                            label="重复惩罚", minimum=1.0, maximum=1.5, value=1.05, step=0.01,
                            info="提高可减少字词重复和机械感，让韵律更丰富。太高(=1.3+)可能导致读音怪异",
                        )
                    with gr.Row():
                        gr.Markdown("##### 🎵 Subtalker（韵律控制）")
                    with gr.Row(elem_classes="aligned-row"):
                        subtalker_temperature = gr.Slider(
                            label="韵律温度", minimum=0.1, maximum=2.0, value=0.9, step=0.1,
                            info="控制语调节奏的随机性。调低=语气更平更稳、语速偏快；调高=语气更生动、抑扬顿挫更丰富",
                        )
                        subtalker_top_k = gr.Slider(
                            label="韵律 Top-K", minimum=1, maximum=100, value=50, step=1,
                            info="控制韵律候选范围。调低 (=20)= 语调变化小、更平淡；调高 (=80)= 语调变化大、更自然",
                        )
                        subtalker_top_p = gr.Slider(
                            label="韵律 Top-P", minimum=0.1, maximum=1.0, value=1.0, step=0.05,
                            info="与韵律 Top-K 配合。调低 (=0.8)= 语调保守；调高 (=1.0)= 语调丰富，建议保持 1.0",
                        )
                    with gr.Row():
                        enable_asr = gr.Checkbox(
                            label="✅ ASR 自动校验（适合隔夜任务）",
                            value=False,
                            info="用 Whisper 识别段头段尾缺字，瑕疵段自动重试（需要 pip install faster-whisper）",
                        )

                # ---- 第四步：分段预览 ----
                with gr.Row():
                    preview_btn = gr.Button("🔍 预览分段", variant="secondary", elem_classes="btn-equal")
                seg_preview = gr.Textbox(
                    label="分段预览", value="点击「预览分段」查看文本分割结果",
                    lines=10, max_lines=20, interactive=False, elem_classes="seg-preview",
                )

                # ---- 第五步：逐段生成与试听 ----
                gr.Markdown("### 🎯 逐段生成与试听")
                gr.Markdown("切换段落选择器可试听已生成的音频；点击「生成/替换该段」可替换该段内容")
                with gr.Row(elem_classes="aligned-row"):
                    segment_selector = gr.Dropdown(
                        label="选择段落", choices=[], interactive=False, scale=2,
                    )
                    gen_one_btn = gr.Button("▶️ 生成/替换该段", variant="secondary", scale=1, elem_classes="btn-equal")
                    gen_all_btn = gr.Button("⏩ 生成所有未生成段", variant="secondary", scale=1, elem_classes="btn-equal")
                with gr.Row():
                    seg_audio_preview = gr.Audio(label="单段试听", type="numpy", interactive=False)
                seg_status = gr.Textbox(
                    label="段落状态", value="⏳ 请先预览分段",
                    lines=8, max_lines=20, interactive=False, elem_classes="seg-preview",
                )

                # ---- 第六步：音频后处理 ----
                gr.Markdown("### 🎛️ 音频后处理")
                gr.Markdown("所有段落生成完成后，在此选择后处理参数并合并输出。合并后分段预览不丢失。")
                with gr.Group():
                    with gr.Row(elem_classes="aligned-row"):
                        speed = gr.Slider(
                            label="语速", minimum=0.5, maximum=1.5, value=0.9, step=0.05,
                            info="0.9=稍慢更自然（推荐），0.7=明显变慢适合长文本，1.0=原速，1.2=偏快",
                        )
                        segment_gap = gr.Slider(
                            label="段间停顿（秒）", minimum=0.0, maximum=5.0, value=1.5, step=0.1,
                            info="段落之间的静音间隔。小说建议 1.0-2.0，文章建议 0.5-1.0",
                        )
                        breathing_pause = gr.Slider(
                            label="气口停顿（秒）", minimum=0.0, maximum=1.0, value=0.25, step=0.05,
                            info="句号/逗号处的短停顿。0=关闭，0.15=紧凑，0.25=自然（推荐），0.4=舒缓",
                        )
                    with gr.Row():
                        merge_btn = gr.Button("🔗 合并已生成段落", variant="primary", size="lg", scale=2, elem_classes="btn-equal")
                    with gr.Row():
                        audio_output = gr.Audio(label="合并结果", type="numpy", interactive=False)

            # =============================================================
            # 右栏：控制台
            # =============================================================
            with gr.Column(scale=1):
                console_output = gr.HTML(value=_build_console_html(), label="控制台")

        # =====================================================================
        # 回调
        # =====================================================================

        # ---- 模型 ----
        load_model_btn.click(
            fn=load_model, inputs=[model_path, device, dtype],
            outputs=[model_status, console_output],
        )
        unload_model_btn.click(
            fn=unload_model,
            outputs=[model_status, console_output],
        )

        # ---- 文件上传 ----
        text_file.upload(
            fn=lambda f: Path(f.name).read_text("utf-8") if f else "",
            inputs=[text_file], outputs=[text_input],
        )

        # ---- 音色 ----
        analyze_btn.click(fn=analyze_voice_profile, inputs=[ref_audio], outputs=[voice_profile])
        load_history_btn.click(
            fn=load_from_history,
            inputs=[ref_history_dd],
            outputs=[ref_audio, ref_text, use_xvec],
        )
        clone_btn.click(
            fn=create_voice_clone,
            inputs=[ref_audio, ref_text, use_xvec, ref_name],
            outputs=[clone_status, ref_history_dd, console_output],
        )

        # ---- 会话管理 ----
        new_session_btn.click(
            fn=do_new_session,
            outputs=[
                seg_preview, segment_selector, seg_status, session_label,
                text_input, seg_audio_preview, console_output,
            ],
        ).then(
            fn=lambda: gr.update(choices=_list_sessions()),
            outputs=[session_selector],
        )
        load_session_btn.click(
            fn=do_load_session,
            inputs=[session_selector],
            outputs=[
                seg_preview, segment_selector, seg_status, session_label,
                text_input, seg_audio_preview, console_output,
            ],
        )

        # ---- 预览分段 ----
        gen_inputs = [text_input, max_seg_len]
        preview_btn.click(
            fn=do_segment, inputs=gen_inputs,
            outputs=[seg_preview, segment_selector, seg_status, console_output],
        )

        # ---- 段落选择器切换 → 自动加载试听 ----
        segment_selector.change(
            fn=on_segment_select,
            inputs=[segment_selector], outputs=[seg_audio_preview],
        )

        # ---- 生成单段 ----
        gen_params = [
            language, max_new_tokens,
            temperature, top_k, top_p, repetition_penalty,
            subtalker_temperature, subtalker_top_k, subtalker_top_p,
        ]
        gen_one_btn.click(
            fn=generate_segment,
            inputs=[segment_selector] + gen_params + [enable_asr],
            outputs=[seg_audio_preview, seg_status, session_label, console_output],
        )

        # ---- 生成所有段 ----
        gen_all_btn.click(
            fn=generate_all_segments,
            inputs=gen_params + [enable_asr],
            outputs=[seg_audio_preview, seg_status, session_label, console_output],
        )

        # ---- 合并 ----
        merge_btn.click(
            fn=merge_segments,
            inputs=[speed, segment_gap, breathing_pause],
            outputs=[audio_output, seg_status, session_label, console_output],
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
