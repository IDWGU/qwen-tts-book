#!/usr/bin/env python3
"""
Qwen3-TTS 有声书生成器 - WebUI (v2)
======================================
磁盘优先架构：所有音频存磁盘，内存零驻留
分步式 Tab 界面 · 段落列表可视化 · 参数预设 · 暗色模式

启动:  python webui.py
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
from long_tts import adjust_speed, insert_breathing_pauses, normalize_audio, split_sentences, level_loudness
from long_tts import init_asr, verify_segment_head_tail

import ref_history

# =====================================================================
# 常量
# =====================================================================

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

PRESETS = {
    "\U0001f3e0 小说朗读": {
        "temperature": 0.8, "top_k": 40, "top_p": 0.95,
        "repetition_penalty": 1.05, "speed": 0.9, "segment_gap": 1.5,
        "subtalker_temperature": 0.8, "subtalker_top_k": 40, "subtalker_top_p": 0.95,
    },
    "\U0001f4f0 新闻播报": {
        "temperature": 0.6, "top_k": 30, "top_p": 0.9,
        "repetition_penalty": 1.02, "speed": 1.0, "segment_gap": 0.8,
        "subtalker_temperature": 0.6, "subtalker_top_k": 30, "subtalker_top_p": 0.9,
    },
    "\U0001f399\ufe0f 情感故事": {
        "temperature": 1.0, "top_k": 60, "top_p": 1.0,
        "repetition_penalty": 1.08, "speed": 0.85, "segment_gap": 2.0,
        "subtalker_temperature": 1.0, "subtalker_top_k": 60, "subtalker_top_p": 1.0,
    },
    "\u26a1 快速生成": {
        "temperature": 0.9, "top_k": 50, "top_p": 1.0,
        "repetition_penalty": 1.05, "speed": 1.0, "segment_gap": 1.0,
        "subtalker_temperature": 0.9, "subtalker_top_k": 50, "subtalker_top_p": 1.0,
    },
}

# =====================================================================
# 全局状态（线程安全，磁盘优先架构）
# =====================================================================

class AppState:
    def __init__(self):
        self._lock = threading.Lock()
        self.tts = None
        self.prompt_items = None
        self.model_loaded = False
        self.model_path = ""
        self.sample_rate = 24000
        self.segments = []                    # 各段文本
        self.segment_files = []               # 各段音频文件路径（唯一内存驻留）
        self.segment_sr = 24000
        self.segment_times = []
        self.source_text = ""
        self.session_name = None
        self.start_time = 0.0
        # ASR 状态（供控制台实时显示）
        self.asr_enabled = False
        self.asr_available = False
        self.asr_passed = 0
        self.asr_retried = 0
        self.asr_failed = 0
        self.asr_current_seg = ""             # 当前正在处理的段描述

    def lock(self):
        return self._lock

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *args):
        self._lock.release()

    @property
    def gen_count(self):
        return sum(1 for f in self.segment_files if f is not None)

    @property
    def gen_total(self):
        return len(self.segments)

    @property
    def elapsed_seconds(self):
        if self.start_time > 0:
            return time.time() - self.start_time
        return 0.0

    @property
    def eta_seconds(self):
        times = [t for t in self.segment_times if t is not None]
        done = len(times)
        total = len(self.segments)
        remaining = total - done
        if times and remaining > 0:
            window = min(len(times), 10)
            avg = sum(times[-window:]) / window
            return avg * remaining
        return None

    def eta_str(self):
        eta = self.eta_seconds
        if eta is not None:
            return f"预计剩余 {int(eta // 60):02d}:{int(eta % 60):02d}"
        return ""


state = AppState()


# =====================================================================
# 会话管理
# =====================================================================

def _list_sessions():
    if not OUTPUTS_DIR.exists():
        return []
    return sorted(
        (d.name for d in OUTPUTS_DIR.iterdir()
         if d.is_dir() and d.name.isdigit() and len(d.name) >= 12),
        reverse=True,
    )


def _now_ts():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def _session_dir(timestamp):
    return OUTPUTS_DIR / timestamp


def _segment_path(session_name, idx):
    sd = _session_dir(session_name)
    sd.mkdir(parents=True, exist_ok=True)
    return str(sd / f"seg_{idx:04d}.wav")


def _get_segment_audio(file_path):
    if file_path and os.path.exists(file_path):
        wav, sr = sf.read(file_path)
        return (sr, wav)
    return None


def _save_session_meta(session_dir, segments, source_text, segment_files, sr, segment_times):
    audio_files = {}
    times = {}
    for i, fpath in enumerate(segment_files):
        if fpath is not None:
            fname = os.path.basename(fpath)
            audio_files[str(i)] = fname
            if segment_times[i] is not None:
                times[str(i)] = segment_times[i]
    meta = {
        "timestamp": session_dir.name,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_text": source_text,
        "segments": segments,
        "audio_files": audio_files,
        "segment_times": times,
        "sample_rate": sr,
    }
    (session_dir / "session.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
    )


def _load_session_meta(timestamp):
    p = _session_dir(timestamp) / "session.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _load_session_paths(timestamp):
    """加载会话 meta + 文件路径列表（不加载音频数据）。"""
    meta = _load_session_meta(timestamp)
    if meta is None:
        return None, 24000, [], []
    sr = meta.get("sample_rate", 24000)
    sd = _session_dir(timestamp)
    audio_files = meta.get("audio_files", {})
    times_raw = meta.get("segment_times", {})
    n = len(meta["segments"])
    files = [None] * n
    times = [None] * n
    for idx_str, fname in audio_files.items():
        idx = int(idx_str)
        fpath = str(sd / fname)
        if os.path.exists(fpath):
            files[idx] = fpath
            if idx_str in times_raw:
                times[idx] = times_raw[idx_str]
    return meta, sr, files, times


def get_session_label():
    with state:
        sn = state.session_name
        cnt = state.gen_count
        tot = state.gen_total
    if not sn:
        return '📁 (无会话)'
    pct = cnt / max(tot, 1) * 100
    return (
        f'<div style="display:flex;align-items:center;gap:8px;font-size:13px;white-space:nowrap;color:var(--body-text-color,#c9d1d9)">'
        f'  <span>📁 {sn}</span>'
        f'  <span style="color:var(--block-label-text-color,#999);font-family:monospace">{cnt}/{tot}</span>'
        f'  <div style="flex:1;height:6px;background:var(--block-border-color,#333);border-radius:3px;overflow:hidden;min-width:60px">'
        f'    <div style="height:100%;width:{pct}%;background:linear-gradient(90deg,#4a9eff,#6c5ce7);border-radius:3px;transition:width .5s"></div>'
        f'  </div>'
        f'</div>'
    )


def _save_current_session():
    if not state.segments or not state.session_name:
        return
    sd = _session_dir(state.session_name)
    _save_session_meta(sd, state.segments, state.source_text,
                        state.segment_files, state.segment_sr,
                        state.segment_times)


# =====================================================================
# HTML 构建器
# =====================================================================

def _build_console_text():
    with state:
        model_ok = state.model_loaded
        voice_ok = state.prompt_items is not None
        gen_cnt = state.gen_count
        seg_cnt = state.gen_total
        times = [t for t in state.segment_times if t is not None]
        elapsed = state.elapsed_seconds
        session_name = state.session_name or "—"
        asr_enabled = state.asr_enabled
        asr_available = state.asr_available
        asr_passed = state.asr_passed
        asr_retried = state.asr_retried
        asr_failed = state.asr_failed
        asr_current = state.asr_current_seg

    elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
    eta_str = state.eta_str()

    model_line = "$ MODEL    " + ("✅ LOADED" if model_ok else "⏳ WAITING")
    voice_line = "$ VOICE    " + ("✅ CREATED" if voice_ok else "⏳ WAITING")
    session_line = f"$ SESSION  {session_name}"

    if seg_cnt > 0:
        blocks_total = 20
        filled = int(gen_cnt / seg_cnt * blocks_total)
        bar = "█" * filled + "░" * (blocks_total - filled)
        pct = f"{gen_cnt / seg_cnt * 100:.0f}"
        progress_line = f"  [{bar}]  {gen_cnt}/{seg_cnt}  {pct}%"
        eta_line = f"  ⏱ {elapsed_str}  {eta_str}" if eta_str else f"  ⏱ {elapsed_str}"
    else:
        progress_line = "  (等待分段)"
        eta_line = ""

    timing_lines = "  ⏱ TIMINGS —"
    if times:
        for i, t in enumerate(times[-5:]):
            idx = len(times) - 5 + i if len(times) > 5 else i
            timing_lines += f'\n  seg_{idx:04d}  {t:.1f}s'
    else:
        timing_lines += "\n  (暂无)"

    asr_lines = ""
    if asr_enabled:
        if asr_available:
            checked = asr_passed + asr_retried + asr_failed
            pct = f"{asr_passed / max(checked, 1) * 100:.0f}%" if checked > 0 else "—"
            asr_lines = f"\n$ ASR      ✅ ACTIVE  ✅{asr_passed}  ↻{asr_retried}  ⚠️{asr_failed}  ({pct})"
            if asr_current:
                asr_lines += f"\n           └ {asr_current}"
        else:
            asr_lines = "\n$ ASR      ❌ UNAVAILABLE (pip install faster-whisper)"

    return "\n".join([
        model_line,
        voice_line,
        session_line,
        asr_lines,
        "",
        "-- PROGRESS --",
        progress_line,
        eta_line,
        "",
        timing_lines,
    ])


def _build_segment_table_html():
    with state:
        segs = list(state.segments)
        files = list(state.segment_files)
        times = list(state.segment_times)

    if not segs:
        return '<div class="st-empty">⏳ 尚未分段，请先执行「预览分段」</div>'

    rows = ""
    for i, seg in enumerate(segs):
        has_audio = files[i] is not None if i < len(files) else False
        t = times[i] if i < len(times) and times[i] is not None else None

        if has_audio:
            icon = "✅"
            cls = "st-done"
            time_str = f"{t:.1f}s" if t else "—"
        else:
            icon = "⏳"
            cls = ""
            time_str = "—"

        preview = seg[:50].replace("\n", "\\n")
        if len(seg) > 50:
            preview += "…"

        rows += (
            f'<div class="st-row {cls}" data-idx="{i}">'
            f'  <span class="st-icon">{icon}</span>'
            f'  <span class="st-num">{i+1}</span>'
            f'  <span class="st-preview">{preview}</span>'
            f'  <span class="st-chars">{len(seg)}字</span>'
            f'  <span class="st-time">{time_str}</span>'
            f'</div>'
        )

    done = sum(1 for f in files[:len(segs)] if f is not None)
    return (
        f'<div class="segment-table">'
        f'  <div class="st-header">📋 段落列表 ({done}/{len(segs)})</div>'
        f'  <div class="st-body">{rows}</div>'
        f'</div>'
    )


# =====================================================================
# 核心逻辑
# =====================================================================

def _resolve_local_cache(repo_id):
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir_name = "models--" + repo_id.replace("/", "--")
    repo_dir = cache_dir / repo_dir_name
    if not repo_dir.exists():
        return None
    refs_dir = repo_dir / "refs"
    if not refs_dir.is_dir():
        return None
    for ref_name in ("main", "master"):
        refs_file = refs_dir / ref_name
        if refs_file.is_file():
            commit_hash = refs_file.read_text().strip()
            snapshot_dir = repo_dir / "snapshots" / commit_hash
            if snapshot_dir.is_dir():
                return str(snapshot_dir)
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


def load_model(model_path, device, dtype_str):
    with state:
        if state.model_loaded and state.model_path == model_path:
            return f"✅ 模型已加载: {model_path}", _build_console_text()
    resolved = _resolve_local_cache(model_path) or model_path
    if resolved != model_path:
        print(f"[load_model] 本地缓存: {resolved}")
    dtype = _DTYPE_MAP.get(dtype_str, torch.float32)
    if device == "mps" and dtype == torch.bfloat16:
        dtype = torch.float32
    attn_impl = "flash_attention_2" if device.startswith("cuda") else None
    try:
        t0 = time.time()
        tts = Qwen3TTSModel.from_pretrained(
            resolved, device_map=device, dtype=dtype,
            attn_implementation=attn_impl, local_files_only=False,
        )
        with state:
            state.tts = tts
            state.model_loaded = True
            state.model_path = model_path
            state.prompt_items = None
        return f"✅ 模型加载完成 ({time.time() - t0:.1f}s)", _build_console_text()
    except Exception as e:
        with state:
            state.model_loaded = False
            state.tts = None
        return f"❌ 模型加载失败: {e}", _build_console_text()


def unload_model():
    with state:
        tts = state.tts
        if tts is None:
            return "⏳ 模型未加载", _build_console_text()
        state.tts = None
        state.prompt_items = None
        state.model_loaded = False
        state.model_path = ""
    try:
        del tts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        return "✅ 模型已卸载", _build_console_text()
    except Exception as e:
        return f"❌ 卸载失败: {e}", _build_console_text()


def load_from_history(selected):
    result = ref_history.resolve(selected)
    if result is None:
        return gr.update(value=None), gr.update(value=""), gr.update(value=False)
    audio_path, text, xvec = result
    return gr.update(value=audio_path), gr.update(value=text), gr.update(value=xvec)


def analyze_voice_profile(audio_path):
    if not audio_path:
        return "⚠️ 请先上传参考音频"
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        if len(y) < sr * 0.5:
            return "⚠️ 音频太短（< 0.5s），建议 3 秒以上"
        rms = librosa.feature.rms(y=y)[0]
        if float(np.mean(rms)) < 1e-8:
            return "⚠️ 音频似乎是静音"
        lines = ["📊 声音特征分析", "=" * 40]
        try:
            f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=2093, sr=sr)
            pitched = f0[voiced] if voiced.any() else np.array([])
            if len(pitched) > 0:
                mp, ps = float(np.mean(pitched)), float(np.std(pitched))
                tags = ["明亮"] if mp > 220 else (["自然"] if mp > 160 else ["沉稳"])
                if ps > 40:
                    tags += ["生动"]
                elif ps > 20:
                    tags += ["自然起伏"]
                else:
                    tags += ["平稳"]
                lines.append(f"🎵 音高: {mp:.0f} Hz")
            else:
                tags = ["自然"]
        except Exception:
            tags = ["自然"]
        mean_v = float(np.mean(rms))
        v_ratio = float(np.std(rms)) / max(mean_v, 1e-10)
        if v_ratio > 1.5:
            lines.append("📢 音量变化: 有爆发力")
        elif v_ratio > 0.8:
            lines.append("📢 音量变化: 有起伏")
        else:
            lines.append("📢 音量变化: 平稳")
        lines.append("")
        lines.append("🎯 推荐风格: " + "、".join(tags))
        lines.append(f"⏱ 时长: {len(y)/sr:.1f}s, 采样率: {sr} Hz")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ 分析失败: {e}"


def create_voice_clone(audio_path, ref_text, use_xvec, ref_name):
    with state as s:
        if not s.model_loaded or s.tts is None:
            return "❌ 请先加载模型", gr.update(), _build_console_text()
        tts = s.tts
    if not audio_path:
        return "❌ 请上传参考音频", gr.update(), _build_console_text()
    try:
        if use_xvec:
            prompt_items = tts.create_voice_clone_prompt(
                ref_audio=audio_path, x_vector_only_mode=True,
            )
        else:
            if not ref_text or not ref_text.strip():
                return "❌ ICL 模式需要参考文本", gr.update(), _build_console_text()
            prompt_items = tts.create_voice_clone_prompt(
                ref_audio=audio_path, ref_text=ref_text.strip(), x_vector_only_mode=False,
            )
        with state:
            state.prompt_items = prompt_items
        if ref_name and ref_name.strip():
            ref_history.add(name=ref_name.strip(), audio_src=audio_path,
                            text=ref_text.strip() if ref_text else "", xvec=use_xvec)
        new_choices = ["-- 新建参考音频 --"] + ref_history.list_names()
        msg = "✅ 音色克隆完成"
        if ref_name and ref_name.strip():
            msg += f"，已保存为「{ref_name.strip()}」"
        return msg, gr.update(choices=new_choices), _build_console_text()
    except Exception as e:
        with state:
            state.prompt_items = None
        return f"❌ 音色克隆失败: {e}", gr.update(), _build_console_text()


# ── 文本分段 ──

def preview_segments(text, max_seg_len):
    if not text or not text.strip():
        return "⚠️ 请输入文本"
    segs = segment_text(text.strip(), max_chars=max_seg_len)
    lines = [f"📄 总字数: {len(text.strip()):,} | 分段数: {len(segs)}"]
    lines.append("=" * 50)
    for i, seg in enumerate(segs):
        p = seg[:80].replace("\n", "\\n")
        if len(seg) > 80:
            p += "..."
        lines.append(f"段[{i+1:03d}] ({len(seg):4d}字): {p}")
    lines.append("=" * 50)
    return "\n".join(lines)


def get_segment_status_display():
    with state:
        if not state.segments:
            return "⏳ 请先分段文本"
        lines = [f"📄 共 {len(state.segments)} 段"]
        for i, seg in enumerate(state.segments):
            has = state.segment_files[i] is not None if i < len(state.segment_files) else False
            icon = "✅" if has else "⏳"
            p = seg[:40].replace("\n", "\\n")
            if len(seg) > 40:
                p += "..."
            lines.append(f"  {icon} 段 [{i+1}] ({len(seg)}字): {p}")
    return "\n".join(lines)


def do_segment(text, max_seg_len):
    if not text or not text.strip():
        with state:
            state.segments = []
            state.segment_files = []
            state.segment_times = []
            state.source_text = ""
        return ("⚠️ 请输入文本",
                gr.update(choices=[], value=None, interactive=False),
                "", get_session_label(), _build_console_text(), _build_segment_table_html())

    segs = segment_text(text.strip(), max_chars=max_seg_len)
    session_name = _now_ts()
    sd = _session_dir(session_name)
    sd.mkdir(parents=True, exist_ok=True)

    with state:
        state.segments = segs
        state.segment_files = [None] * len(segs)
        state.segment_times = [None] * len(segs)
        state.segment_sr = 24000
        state.source_text = text.strip()
        state.session_name = session_name
        state.start_time = time.time()

    preview = preview_segments(text, max_seg_len)
    choices = [f"段 {i+1} ({len(seg)}字)" for i, seg in enumerate(segs)]
    status = get_segment_status_display()
    return (preview,
            gr.update(choices=choices, value=choices[0] if choices else None, interactive=True),
            status, get_session_label(), _build_console_text(), _build_segment_table_html())


# ── 内存清理 ──

def _cleanup_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()


# ── Token 预算 ──

def _calc_max_tokens(text_len: int) -> int:
    """根据文本字符数自动计算需要的 max_new_tokens（Qwen3-TTS 12Hz）。"""
    # 每个中文字符大约需要 40 个音频 token（含停顿、气口、韵律余量）
    # 下限 512，上限 32768（≈45 分钟音频）
    return min(max(int(text_len * 40), 512), 32768)


# ── 逐段生成（磁盘优先 + 内存清理） ──

def generate_segment(
    seg_label,
    temperature, top_k, top_p, repetition_penalty,
    subtalker_temperature, subtalker_top_k, subtalker_top_p,
    enable_asr=False,
):
    with state as s:
        if not s.model_loaded or s.tts is None:
            return gr.update(), "❌ 请先加载模型", get_session_label(), _build_console_text(), _build_segment_table_html()
        if s.prompt_items is None:
            return gr.update(), "❌ 请先创建音色克隆", get_session_label(), _build_console_text(), _build_segment_table_html()
        if not s.segments:
            return gr.update(), "❌ 请先分段文本", get_session_label(), _build_console_text(), _build_segment_table_html()
        tts = s.tts
        prompt_items = s.prompt_items
        segments = list(s.segments)
        session_name = s.session_name

    if not seg_label:
        return gr.update(), "❌ 请选择段落", get_session_label(), _build_console_text(), _build_segment_table_html()

    try:
        seg_idx = int(seg_label.split()[1]) - 1
    except (ValueError, IndexError):
        return gr.update(), "❌ 无效选择", get_session_label(), _build_console_text(), _build_segment_table_html()
    if seg_idx < 0 or seg_idx >= len(segments):
        return gr.update(), "❌ 段号范围错误", get_session_label(), _build_console_text(), _build_segment_table_html()

    seg_text = segments[seg_idx]
    t0 = time.time()
    gen_kwargs = dict(
        max_new_tokens=_calc_max_tokens(len(seg_text)), do_sample=True,
        top_k=top_k, top_p=top_p, temperature=temperature,
        repetition_penalty=repetition_penalty,
        subtalker_dosample=True, subtalker_top_k=subtalker_top_k,
        subtalker_top_p=subtalker_top_p, subtalker_temperature=subtalker_temperature,
    )

    asr_available = init_asr(quiet=True) if enable_asr else False
    with state:
        state.asr_enabled = enable_asr
        state.asr_available = asr_available
        state.asr_passed = 0
        state.asr_retried = 0
        state.asr_failed = 0
        state.asr_current_seg = f"段 [{seg_idx+1}] 合成中..."
    retry_kwargs = dict(gen_kwargs)
    max_retry = 3 if enable_asr else 0
    best_wav, best_sr_val = None, None

    for attempt in range(max_retry + 1):
        try:
            wavs, sr_val = tts.generate_voice_clone(
                text=seg_text, language="Auto",
                voice_clone_prompt=prompt_items, **retry_kwargs,
            )
            seg_wav = wavs[0]
            seg_wav = np.clip(seg_wav, -1.0, 1.0)

            # 立即写盘，不驻留内存
            fpath = _segment_path(session_name, seg_idx)
            sf.write(fpath, seg_wav, sr_val)

            if not enable_asr or not asr_available:
                elapsed = time.time() - t0
                with state:
                    state.segment_files[seg_idx] = fpath
                    state.segment_sr = sr_val
                    if seg_idx < len(state.segment_times):
                        state.segment_times[seg_idx] = elapsed
                    state.asr_current_seg = ""
                del seg_wav, wavs
                _cleanup_gpu_memory()
                _save_current_session()
                return (gr.update(value=fpath),
                        f"✅ 段 [{seg_idx+1}] 完成\n{get_segment_status_display()}",
                        get_session_label(), _build_console_text(), _build_segment_table_html())

            # ASR
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            sf.write(tmp_path, seg_wav, sr_val)
            try:
                passed, _, _, fail_reason = verify_segment_head_tail(
                    tmp_path, seg_text, language="Auto",
                )
            finally:
                os.unlink(tmp_path)

            if passed:
                elapsed = time.time() - t0
                with state:
                    state.segment_files[seg_idx] = fpath
                    state.segment_sr = sr_val
                    if seg_idx < len(state.segment_times):
                        state.segment_times[seg_idx] = elapsed
                    state.asr_current_seg = ""
                del seg_wav, wavs
                _cleanup_gpu_memory()
                _save_current_session()
                return (gr.update(value=fpath),
                        f"✅ 段 [{seg_idx+1}] 完成 ✅ ASR✓\n{get_segment_status_display()}",
                        get_session_label(), _build_console_text(), _build_segment_table_html())
            else:
                if attempt == 0:
                    best_wav, best_sr_val = seg_wav.copy(), sr_val
                if attempt < max_retry:
                    retry_kwargs["temperature"] = min(retry_kwargs["temperature"] + 0.1, 1.2)
                    del seg_wav, wavs
                    _cleanup_gpu_memory()
                else:
                    elapsed = time.time() - t0
                    with state:
                        state.segment_files[seg_idx] = fpath
                        state.segment_sr = best_sr_val or sr_val
                        if seg_idx < len(state.segment_times):
                            state.segment_times[seg_idx] = elapsed
                        state.asr_current_seg = ""
                    del seg_wav, wavs, best_wav
                    _cleanup_gpu_memory()
                    _save_current_session()
                    return (gr.update(value=fpath),
                            f"⚠️ 段 [{seg_idx+1}] ASR 校验失败，已保留\n{get_segment_status_display()}",
                            get_session_label(), _build_console_text(), _build_segment_table_html())

        except Exception as e:
            with state:
                state.asr_current_seg = ""
            _cleanup_gpu_memory()
            return (gr.update(),
                    f"❌ 段 [{seg_idx+1}] 失败: {e}\n{get_segment_status_display()}",
                    get_session_label(), _build_console_text(), _build_segment_table_html())

    with state:
        state.asr_current_seg = ""
    return gr.update(), "❌ 未知错误", get_session_label(), _build_console_text(), _build_segment_table_html()


# ── 全部生成（磁盘优先 + 内存清理） ──

def generate_all_segments(
    temperature, top_k, top_p, repetition_penalty,
    subtalker_temperature, subtalker_top_k, subtalker_top_p,
    enable_asr=False,
):
    """全部生成（generator，yield 实时更新控制台）。"""
    with state as s:
        if not s.model_loaded or s.tts is None:
            yield gr.update(), "❌ 请先加载模型", get_session_label(), _build_console_text(), _build_segment_table_html()
            return
        if s.prompt_items is None:
            yield gr.update(), "❌ 请先创建音色", get_session_label(), _build_console_text(), _build_segment_table_html()
            return
        if not s.segments:
            yield gr.update(), "❌ 请先分段", get_session_label(), _build_console_text(), _build_segment_table_html()
            return
        tts = s.tts
        prompt_items = s.prompt_items
        segments = list(s.segments)
        total = len(segments)
        session_name = s.session_name
        needed = [i for i, f in enumerate(s.segment_files) if f is None]

    if not needed:
        yield (gr.update(),
                "✅ 所有段落已生成\n\n" + get_segment_status_display(),
                get_session_label(), _build_console_text(), _build_segment_table_html())
        return

    asr_available = init_asr(quiet=True) if enable_asr else False
    base_gen_kwargs = dict(
        do_sample=True,
        top_k=top_k, top_p=top_p, temperature=temperature,
        repetition_penalty=repetition_penalty,
        subtalker_dosample=True, subtalker_top_k=subtalker_top_k,
        subtalker_top_p=subtalker_top_p, subtalker_temperature=subtalker_temperature,
    )
    last_audio_path = None
    errors = 0
    need_count = len(needed)
    verify_passed = 0
    verify_retried = 0
    verify_failed = 0

    # 初始化控制台 ASR 状态
    with state:
        state.asr_enabled = enable_asr
        state.asr_available = asr_available
        state.asr_passed = 0
        state.asr_retried = 0
        state.asr_failed = 0

    for seq_idx, i in enumerate(needed):
        seg_text = segments[i]
        t0 = time.time()
        retry_kwargs = dict(base_gen_kwargs, max_new_tokens=_calc_max_tokens(len(seg_text)))
        max_retry = 3 if enable_asr and asr_available else 0
        seg_success = False

        with state:
            state.asr_current_seg = f"段 [{i+1}/{total}] 合成中..."

        for attempt in range(max_retry + 1):
            try:
                with state:
                    state.asr_current_seg = f"段 [{i+1}/{total}] 合成中 (尝试 {attempt+1})..."
                wavs, sr_local = tts.generate_voice_clone(
                    text=seg_text, language="Auto",
                    voice_clone_prompt=prompt_items, **retry_kwargs,
                )
                seg_wav = wavs[0]
                seg_wav = np.clip(seg_wav, -1.0, 1.0)

                # 写盘
                fpath = _segment_path(session_name, i)
                sf.write(fpath, seg_wav, sr_local)

                if not enable_asr or not asr_available:
                    elapsed = time.time() - t0
                    with state:
                        state.segment_files[i] = fpath
                        state.segment_sr = sr_local
                        if i < len(state.segment_times):
                            state.segment_times[i] = elapsed
                        state.asr_current_seg = f"段 [{i+1}/{total}] ✅ {elapsed:.1f}s"
                    last_audio_path = fpath
                    seg_success = True
                    del seg_wav, wavs
                    _cleanup_gpu_memory()
                    break

                # ASR
                with state:
                    state.asr_current_seg = f"段 [{i+1}/{total}] ASR 校验中..."
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_path = tmp.name
                tmp.close()
                sf.write(tmp_path, seg_wav, sr_local)
                try:
                    passed, _, _, fail_reason = verify_segment_head_tail(
                        tmp_path, seg_text, language="Auto",
                    )
                finally:
                    os.unlink(tmp_path)

                if passed:
                    verify_passed += 1
                    with state:
                        state.asr_passed = verify_passed
                    elapsed = time.time() - t0
                    with state:
                        state.segment_files[i] = fpath
                        state.segment_sr = sr_local
                        if i < len(state.segment_times):
                            state.segment_times[i] = elapsed
                        state.asr_current_seg = f"段 [{i+1}/{total}] ✅ ASR✓ {elapsed:.1f}s"
                    last_audio_path = fpath
                    seg_success = True
                    del seg_wav, wavs
                    _cleanup_gpu_memory()
                    break
                else:
                    if attempt < max_retry:
                        verify_retried += 1
                        with state:
                            state.asr_retried = verify_retried
                            state.asr_current_seg = f"段 [{i+1}/{total}] ASR 失败，重试 {attempt+1}/{max_retry}"
                        retry_kwargs["temperature"] = min(
                            retry_kwargs["temperature"] + 0.1, 1.2)
                        del seg_wav, wavs
                        _cleanup_gpu_memory()
                    else:
                        verify_failed += 1
                        with state:
                            state.asr_failed = verify_failed
                        elapsed = time.time() - t0
                        with state:
                            state.segment_files[i] = fpath
                            state.segment_sr = sr_local
                            if i < len(state.segment_times):
                                state.segment_times[i] = elapsed
                            state.asr_current_seg = f"段 [{i+1}/{total}] ⚠️ ASR 失败，已保留"
                        last_audio_path = fpath
                        seg_success = True
                        del seg_wav, wavs
                        _cleanup_gpu_memory()
                        break

            except Exception:
                _cleanup_gpu_memory()
                break

        if not seg_success:
            errors += 1
            with state:
                state.asr_current_seg = f"段 [{i+1}/{total}] ❌ 失败"

        # 每段完成后 yield 更新 UI（控制台实时刷新）
        status_text = get_segment_status_display()
        yield (gr.update(value=last_audio_path),
               status_text,
               get_session_label(), _build_console_text(), _build_segment_table_html())

    # 清除当前段状态
    with state:
        state.asr_current_seg = ""

    _save_current_session()

    # 最终总结
    with state:
        done = state.gen_count

    lines = [f"✅ 已完成 {done}/{total} 段"]
    if errors:
        lines.append(f" ({errors} 段失败)")

    if enable_asr and asr_available:
        checked = verify_passed + verify_retried + verify_failed
        if checked > 0:
            pct = verify_passed / max(checked, 1) * 100
            lines.append(f"\n📊 ASR: {checked} 段")
            lines.append(f"   ✅ 一次通过: {verify_passed} ({pct:.0f}%)")
            if verify_retried:
                lines.append(f"   ↻ 重试: {verify_retried}")
            if verify_failed:
                lines.append(f"   ⚠️ 失败: {verify_failed}（已保留）")

    if done == total and not errors:
        lines.append("\n💡 全部完成！前往「合并」Tab 输出最终音频。")
    else:
        lines.append("\n部分段未完成，可继续生成")

    yield (gr.update(value=last_audio_path),
            "\n".join(lines) + "\n" + get_segment_status_display(),
            get_session_label(), _build_console_text(), _build_segment_table_html())


# ── 合并（从磁盘读取） ──

def merge_segments(speed, segment_gap, breathing_pause, enable_leveling, leveling_strength):
    with state as s:
        if not s.segment_files or all(f is None for f in s.segment_files):
            return gr.update(), "❌ 没有已生成的段落", get_session_label(), _build_console_text(), _build_segment_table_html()
        generated = [(i, f) for i, f in enumerate(s.segment_files) if f is not None]
        total = len(s.segment_files)
        done = len(generated)
        sr = s.segment_sr
        segs_txt = [s.segments[i] for i, _ in generated]

    warn = ""
    if done < total:
        if done == 0:
            return gr.update(), "❌ 所有段落均未生成", get_session_label(), _build_console_text(), _build_segment_table_html()
        warn = f"⚠️ 还有 {total - done} 段未生成，仅合并 {done}/{total} 段\n\n"

    # 按需从磁盘读取
    pwavs = []
    for (i, fpath), stxt in zip(generated, segs_txt):
        wav, local_sr = sf.read(fpath)
        # 后处理
        wav = normalize_audio(wav, local_sr or sr)
        # 响度均衡（消除段内前后音量忽大忽小）
        if enable_leveling:
            wav = level_loudness(wav, local_sr or sr, strength=leveling_strength)
        if breathing_pause > 0:
            wav = insert_breathing_pauses(wav, stxt, local_sr or sr, breathing_pause)
        pwavs.append((wav, local_sr or sr))

    merged_wav, sr_out = merge_audio_segments(pwavs, segment_gap=segment_gap)

    # 释放合并用的中间列表
    del pwavs
    gc.collect()

    if abs(speed - 1.0) > 0.01:
        merged_wav = adjust_speed(merged_wav, sr_out, speed)

    # 保存到会话目录
    merged_path = ""
    with state:
        if state.session_name:
            sd = _session_dir(state.session_name)
            sd.mkdir(parents=True, exist_ok=True)
            mf = sd / "merged.wav"
            sf.write(str(mf), merged_wav, sr_out)
            merged_path = str(mf)
            # 更新 session.json
            meta_path = sd / "session.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["merged_file"] = "merged.wav"
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    total_dur = len(merged_wav) / sr_out
    del merged_wav
    gc.collect()

    path_info = f"\n   📁 已保存: {merged_path}" if merged_path else ""
    status = (
        f"{warn}🔗 合并完成! {total_dur:.1f}s ({total_dur/60:.1f} 分)"
        f"{path_info}\n"
        f"   不满意可重新生成某段后再点合并\n"
        f"{get_segment_status_display()}"
    )
    return gr.update(value=merged_path), status, get_session_label(), _build_console_text(), _build_segment_table_html()


# ── 会话操作 ──

def do_new_session():
    with state:
        state.segments = []
        state.segment_files = []
        state.segment_times = []
        state.source_text = ""
        state.session_name = _now_ts()
        state.start_time = 0
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = _list_sessions()
    return ("📭 新会话已创建",
            gr.update(choices=[], value=None, interactive=False),
            "⏳ 等待分段", get_session_label(), "", None,
            _build_console_text(), _build_segment_table_html(),
            gr.update(choices=sessions))


def do_load_session(timestamp):
    if not timestamp:
        return ("⚠️ 请选择会话", gr.update(), get_session_label(), "", "",
                None, _build_console_text(), _build_segment_table_html())

    meta, sr, files, times = _load_session_paths(timestamp)
    if meta is None:
        return (f"❌ 会话 {timestamp} 不存在", gr.update(), get_session_label(), "", "",
                None, _build_console_text(), _build_segment_table_html())

    with state:
        state.segments = meta["segments"]
        state.segment_files = files
        state.segment_sr = sr
        state.source_text = meta.get("source_text", "")
        state.session_name = timestamp
        state.start_time = time.time()
        state.segment_times = times

    preview = preview_segments(state.source_text or " ".join(state.segments), 500)
    choices = [f"段 {i+1} ({len(seg)}字)" for i, seg in enumerate(state.segments)]
    return (preview,
            gr.update(choices=choices, value=choices[0] if choices else None, interactive=True),
            get_segment_status_display(), get_session_label(), state.source_text,
            None, _build_console_text(), _build_segment_table_html())


def on_segment_select(seg_label):
    """段选择器切换时，从磁盘读取音频播放。"""
    if not seg_label or not state.segments:
        return gr.update(value=None)
    try:
        idx = int(seg_label.split()[1]) - 1
    except (ValueError, IndexError):
        return gr.update(value=None)
    with state:
        if 0 <= idx < len(state.segment_files) and state.segment_files[idx] is not None:
            return gr.update(value=state.segment_files[idx])
    return gr.update(value=None)


# =====================================================================
# CSS 样式
# =====================================================================

CSS = """
.gradio-container { max-width: 1480px !important; }
footer { display: none !important; }

/* ── 顶部状态条 ── */
.status-bar { margin-bottom: 8px; }
.status-box { margin-bottom: 0 !important; min-height: 32px; }

/* ── 段落列表（主题自适应） ── */
.segment-table {
    border: 1px solid var(--block-border-color, #30363d); border-radius: 6px; overflow: hidden;
    max-height: 360px; overflow-y: auto; background: var(--block-background-fill, #0d1117);
}
.st-header {
    background: var(--background-fill-primary, #161b22); padding: 8px 12px; font-size: 13px; font-weight: 600;
    border-bottom: 1px solid var(--block-border-color, #30363d); position: sticky; top: 0; z-index: 1;
    color: var(--body-text-color, #c9d1d9);
}
.st-empty { padding: 20px; text-align: center; color: var(--block-label-text-color, #484f58); font-size: 13px; }
.st-body { }
.st-row {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 12px; font-size: 12px; border-bottom: 1px solid var(--block-border-color, #21262d);
    transition: background 0.15s; color: var(--body-text-color-secondary, #8b949e);
}
.st-row:hover { background: var(--block-background-fill-hover, #1c2128); }
.st-row.st-done { color: var(--body-text-color, #c9d1d9); }
.st-icon { width: 20px; text-align: center; font-size: 14px; }
.st-num { width: 28px; color: var(--block-label-text-color, #484f58); font-family: monospace; font-size: 11px; }
.st-preview { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.st-chars { width: 48px; text-align: right; color: var(--block-label-text-color, #484f58); font-size: 11px; }
.st-time { width: 48px; text-align: right; color: var(--block-label-text-color, #484f58); font-family: monospace; font-size: 11px; }

/* ── 预设 dropdown 高亮 ── */
#preset-dropdown { margin-bottom: 4px; }
/* ── 文本区域统一 ── */
.seg-preview { font-family: 'SF Mono','Menlo',monospace; font-size: 13px; }
"""


# =====================================================================
# UI 构建
# =====================================================================

def build_ui():
    with gr.Blocks(title="Qwen3-TTS 有声书生成器") as demo:
        gr.Markdown(
            "# 🎧 Qwen3-TTS 有声书生成器 v2\n"
            "磁盘优先架构 · 内存零驻留 · 分步式操作"
        )

        # ── 顶部状态条 ──
        with gr.Row(elem_classes="status-bar"):
            model_status = gr.Textbox(label="🤖 模型", value="⏳ 等待加载",
                                       scale=1, elem_classes="status-box")
            voice_status = gr.Textbox(label="🎤 音色", value="⏳ 等待创建",
                                       scale=1, elem_classes="status-box")
            session_label_out = gr.HTML(label="📁 会话",
                                         value=get_session_label(),
                                         elem_classes="status-box")

        # ── 主体：左（操作区）+ 右（控制台） ──
        with gr.Row():
            with gr.Column(scale=3):
                # == 会话管理（始终可见） ==
                with gr.Row():
                    with gr.Column(scale=1):
                        session_selector = gr.Dropdown(
                            label="📂 历史会话", choices=_list_sessions(),
                            interactive=True, scale=3,
                        )
                with gr.Row():
                    load_session_btn = gr.Button("📥 加载", variant="secondary", scale=1)
                    new_session_btn = gr.Button("📄 新建", variant="secondary", scale=1)


                # == Tab 分步 ==
                with gr.Tabs():
                    # =============================================================
                    # Tab 1: 音色
                    # =============================================================
                    with gr.TabItem("🎤 音色"):
                        with gr.Accordion("⚙️ 模型设置", open=False):
                            with gr.Row():
                                model_path = gr.Textbox(
                                    label="模型路径", scale=3,
                                    value="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                )
                                device = gr.Dropdown(label="设备",
                                    choices=["mps", "cpu", "cuda:0"], value="mps", scale=1)
                                dtype = gr.Dropdown(label="精度",
                                    choices=["float32", "float16", "bfloat16"],
                                    value="float16", scale=1)
                        # 加载/卸载按钮放在 Accordion 外部，确保 Gradio 6.0 事件绑定正常
                        with gr.Row():
                            load_model_btn = gr.Button("🚀 加载模型", variant="primary", scale=1)
                            unload_model_btn = gr.Button("⏏️ 卸载", variant="secondary", scale=1)

                        with gr.Group():
                            gr.Markdown("### 🎤 音色克隆")
                            with gr.Row():
                                ref_history_dd = gr.Dropdown(
                                    label="📂 历史音频",
                                    choices=["-- 新建参考音频 --"] + ref_history.list_names(),
                                    value="-- 新建参考音频 --",
                                    interactive=True, scale=2,
                                )
                                ref_name = gr.Textbox(label="保存名称", scale=2,
                                    placeholder="填写后自动保存历史")
                                load_history_btn = gr.Button("📂 加载", variant="secondary", scale=1)
                            with gr.Row():
                                ref_audio = gr.Audio(label="参考音频（录制或上传，3s+）",
                                                     type="filepath")
                                ref_text = gr.Textbox(
                                    label="参考文字（ICL 模式）", lines=3,
                                    placeholder="输入参考音频的说话内容...",
                                )
                            with gr.Row():
                                use_xvec = gr.Checkbox(label="x-vector 模式（无需参考文本）", value=False)
                                analyze_btn = gr.Button("🔍 声音分析", variant="secondary", scale=1)
                                clone_btn = gr.Button("🎯 创建音色", variant="primary", scale=1)
                            voice_profile = gr.Textbox(label="声音特征", lines=5, max_lines=15,
                                                        interactive=False)

                    # =============================================================
                    # Tab 2: 文本
                    # =============================================================
                    with gr.TabItem("📝 文本"):
                        with gr.Row():
                            text_input = gr.Textbox(label="待合成文本", lines=12,
                                                     placeholder="粘贴或输入长文本...", scale=3)
                            text_file = gr.File(label="或上传 .txt", file_types=[".txt"], scale=1)
                        with gr.Row():
                            max_seg_len = gr.Slider(
                                label="每段字符", minimum=50, maximum=2000,
                                value=200, step=50,
                                info="中文建议 300-500，英文 500-1000；max_new_tokens 自动根据此值与文本长度计算",
                            )
                        with gr.Row():
                            preview_btn = gr.Button("🔍 预览分段", variant="secondary", scale=1)
                        seg_preview = gr.Textbox(
                            label="分段预览", value="点击「预览分段」查看结果",
                            lines=8, max_lines=16, interactive=False,
                            elem_classes="seg-preview",
                        )

                    # =============================================================
                    # Tab 3: 生成
                    # =============================================================
                    with gr.TabItem("🎯 生成"):
                        # 段落概览
                        segment_table = gr.HTML(value=_build_segment_table_html(),
                                                 label="段落列表")

                        # 段落选择 + 操作
                        with gr.Row():
                            segment_selector = gr.Dropdown(
                                label="选择段落", choices=[], interactive=False, scale=3,
                            )
                            gen_one_btn = gr.Button("▶️ 生成该段", variant="secondary", scale=1)
                            gen_all_btn = gr.Button("⏩ 生成所有", variant="secondary", scale=1)

                        # 试听
                        seg_audio_preview = gr.Audio(label="单段试听", type="filepath", interactive=False)
                        seg_status = gr.Textbox(
                            label="段落状态", value="⏳ 请先预览分段",
                            lines=4, max_lines=12, interactive=False,
                            elem_classes="seg-preview",
                        )

                        # 参数（可折叠）
                        with gr.Accordion("🔧 生成参数", open=False):
                            preset_dd = gr.Dropdown(
                                label="🎨 参数预设（快速切换参数组合）",
                                choices=["— 手动调节 —"] + list(PRESETS.keys()),
                                value="— 手动调节 —",
                                interactive=True,
                                elem_id="preset-dropdown",
                            )
                            with gr.Row():
                                temperature = gr.Slider(label="温度", minimum=0.1, maximum=2.0,
                                                         value=0.9, step=0.1)
                                top_k = gr.Slider(label="Top-K", minimum=1, maximum=100,
                                                    value=50, step=1)
                                top_p = gr.Slider(label="Top-P", minimum=0.1, maximum=1.0,
                                                   value=1.0, step=0.05)
                                repetition_penalty = gr.Slider(label="重复惩罚", minimum=1.0, maximum=1.5,
                                                                value=1.05, step=0.01)
                            gr.Markdown("##### 🎵 Subtalker（韵律）")
                            with gr.Row():
                                subtalker_temperature = gr.Slider(label="韵律温度", minimum=0.1, maximum=2.0,
                                                                   value=0.9, step=0.1)
                                subtalker_top_k = gr.Slider(label="韵律 Top-K", minimum=1, maximum=100,
                                                              value=50, step=1)
                                subtalker_top_p = gr.Slider(label="韵律 Top-P", minimum=0.1, maximum=1.0,
                                                             value=1.0, step=0.05)
                            with gr.Row():
                                enable_asr = gr.Checkbox(
                                    label="ASR 自动校验（隔夜任务推荐）", value=False,
                                    info="需要 pip install faster-whisper",
                                )

                    # =============================================================
                    # Tab 4: 合并
                    # =============================================================
                    with gr.TabItem("🎛️ 合并"):
                        gr.Markdown("所有段落生成后，在此调整参数后合并输出。")
                        with gr.Row():
                            speed = gr.Slider(label="语速", minimum=0.5, maximum=1.5,
                                               value=0.9, step=0.05,
                                               info="0.9=稍慢自然（推荐）")
                            segment_gap = gr.Slider(label="段间停顿(秒)", minimum=0.0, maximum=5.0,
                                                     value=1.5, step=0.1,
                                                     info="小说 1.0-2.0")
                            breathing_pause = gr.Slider(label="气口停顿(秒)", minimum=0.0, maximum=1.0,
                                                         value=0.25, step=0.05,
                                                         info="0.25=自然（推荐）")
                        with gr.Row():
                            enable_leveling = gr.Checkbox(
                                label="🎚️ 音量均衡（消除前后忽大忽小）", value=True,
                                info="自动调整每段内音量，使整篇响度一致。建议开启。",
                            )
                            leveling_strength = gr.Slider(
                                label="均衡力度", minimum=0.0, maximum=1.0,
                                value=0.8, step=0.05,
                                info="0.0=无效果, 0.5=半程修正, 1.0=完全均衡",
                            )
                        with gr.Row():
                            merge_btn = gr.Button("🔗 合并并输出", variant="primary",
                                                   size="lg", scale=2)
                        audio_output = gr.Audio(label="合并结果", type="filepath", interactive=False)

            # =============================================================
            # 右栏：控制台
            # =============================================================
            with gr.Column(scale=1):
                console_output = gr.Textbox(value=_build_console_text(), label="控制台", lines=15)

        # =====================================================================
        # 回调绑定
        # =====================================================================

        # ── 模型 ──
        load_model_btn.click(
            fn=load_model, inputs=[model_path, device, dtype],
            outputs=[model_status, console_output],
        )
        unload_model_btn.click(
            fn=unload_model,
            outputs=[model_status, console_output],
        )

        # ── 文件上传 ──
        text_file.upload(
            fn=lambda f: Path(f.name).read_text("utf-8") if f else "",
            inputs=[text_file], outputs=[text_input],
        )

        # ── 音色 ──
        analyze_btn.click(fn=analyze_voice_profile, inputs=[ref_audio],
                          outputs=[voice_profile])
        load_history_btn.click(
            fn=load_from_history,
            inputs=[ref_history_dd],
            outputs=[ref_audio, ref_text, use_xvec],
        )
        clone_btn.click(
            fn=create_voice_clone,
            inputs=[ref_audio, ref_text, use_xvec, ref_name],
            outputs=[voice_status, ref_history_dd, console_output],
        )

        # ── 会话 ──
        new_session_btn.click(
            fn=do_new_session,
            outputs=[seg_preview, segment_selector, seg_status, session_label_out,
                     text_input, seg_audio_preview, console_output, segment_table,
                     session_selector],
        )
        load_session_btn.click(
            fn=do_load_session,
            inputs=[session_selector],
            outputs=[seg_preview, segment_selector, seg_status, session_label_out,
                     text_input, seg_audio_preview, console_output, segment_table],
        )

        # ── 文本分段 ──
        preview_btn.click(
            fn=do_segment, inputs=[text_input, max_seg_len],
            outputs=[seg_preview, segment_selector, seg_status, session_label_out, console_output, segment_table],
        )

        # ── 段落选择 → 自动加载音频 ──
        segment_selector.change(
            fn=on_segment_select,
            inputs=[segment_selector], outputs=[seg_audio_preview],
        )

        # ── 生成 ──
        gen_params = [
            temperature, top_k, top_p, repetition_penalty,
            subtalker_temperature, subtalker_top_k, subtalker_top_p,
        ]
        gen_one_btn.click(
            fn=generate_segment,
            inputs=[segment_selector] + gen_params + [enable_asr],
            outputs=[seg_audio_preview, seg_status, session_label_out, console_output, segment_table],
        )
        gen_all_btn.click(
            fn=generate_all_segments,
            inputs=gen_params + [enable_asr],
            outputs=[seg_audio_preview, seg_status, session_label_out, console_output, segment_table],
        )

        # ── 合并 ──
        merge_btn.click(
            fn=merge_segments,
            inputs=[speed, segment_gap, breathing_pause, enable_leveling, leveling_strength],
            outputs=[audio_output, seg_status, session_label_out, console_output, segment_table],
        )

        # ── 预设切换 ──
        # 预设下拉本身不触发参数更新，我们需要一个隐藏组件来接收 select 事件
        # Gradio Dropdown 的 select 事件直接触发
        def _apply_preset(preset_name):
            if preset_name not in PRESETS:
                return [gr.update()] * 9
            p = PRESETS[preset_name]
            return [
                gr.update(value=p["temperature"]),
                gr.update(value=p["top_k"]),
                gr.update(value=p["top_p"]),
                gr.update(value=p["repetition_penalty"]),
                gr.update(value=p["subtalker_temperature"]),
                gr.update(value=p["subtalker_top_k"]),
                gr.update(value=p["subtalker_top_p"]),
                gr.update(value=p["speed"]),
                gr.update(value=p["segment_gap"]),
            ]

        preset_dd.change(
            fn=_apply_preset,
            inputs=[preset_dd],
            outputs=[temperature, top_k, top_p, repetition_penalty,
                     subtalker_temperature, subtalker_top_k, subtalker_top_p,
                     speed, segment_gap],
        )

    return demo


# =====================================================================
# 启动
# =====================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-TTS WebUI v2")
    parser.add_argument("--ip", default="127.0.0.1", help="监听地址 (默认: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="监听端口 (默认: 7860)")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    args = parser.parse_args()

    demo = build_ui()
    print(f"\n🌐 启动 WebUI v2: http://{args.ip}:{args.port}")
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=CSS,
        theme=gr.themes.Soft(),
    )

