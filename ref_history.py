#!/usr/bin/env python3
"""
参考音频历史记录模块
=====================
存储/读取参考音频及其文字内容的记录，方便跨会话复用。

存储位置：项目根目录
  - ref_history.json   索引文件
  - ref_history/        音频文件目录
"""

import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_DIR = Path(__file__).parent
HISTORY_FILE = PROJECT_DIR / "ref_history.json"
HISTORY_DIR = PROJECT_DIR / "ref_history"


def load_all() -> List[Dict]:
    """加载所有历史条目。

    Returns:
        [{"id", "name", "audio", "text", "xvec", "created"}, ...]
    """
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text("utf-8"))
        return data.get("entries", [])
    except (json.JSONDecodeError, OSError):
        return []


def save_all(entries: List[Dict]):
    """保存所有历史条目。"""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(
        json.dumps({"entries": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add(
    name: str,
    audio_src: str,
    text: str,
    xvec: bool = False,
) -> Dict:
    """添加一条历史记录。

    Args:
        name: 用户自定义名称（可为空，自动用 id 代替）
        audio_src: 源音频文件路径
        text: 参考文本
        xvec: 是否 x-vector 模式

    Returns:
        新创建的条目 dict
    """
    entries = load_all()
    entry_id = time.strftime("%Y%m%d_%H%M%S")

    # 复制音频到历史目录
    ext = Path(audio_src).suffix or ".wav"
    audio_dst = str(HISTORY_DIR / f"{entry_id}{ext}")
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(audio_src, audio_dst)

    entry = {
        "id": entry_id,
        "name": name if name.strip() else entry_id,
        "audio": audio_dst,
        "text": text,
        "xvec": xvec,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    entries.append(entry)
    save_all(entries)
    return entry


def find(identifier: str) -> Optional[Dict]:
    """按 id 或 name 查找条目。"""
    for e in load_all():
        if e["id"] == identifier or e["name"] == identifier:
            return e
    return None


def delete(identifier: str) -> bool:
    """删除一条历史记录。"""
    entries = load_all()
    before = len(entries)
    entries = [e for e in entries if e["id"] != identifier and e["name"] != identifier]
    if len(entries) == before:
        return False
    save_all(entries)
    return True


def list_names() -> List[str]:
    """返回 [name (时间), ...] 格式的列表，用于 dropdown 选项。

    第一个固定为 "-- 新建参考音频 --"，表示不使用历史。
    """
    items = []
    for e in load_all():
        items.append(f"{e['name']} ({e['created']})")
    return items


def resolve(name_str: str) -> Optional[Tuple[str, str, bool]]:
    """将 dropdown 选中项解析为 (audio_path, text, xvec)。

    Args:
        name_str: format "name (created_at)"

    Returns:
        (audio_path, text, xvec) or None if not found
    """
    if not name_str or name_str.startswith("--"):
        return None
    # 反向查找：匹配 name 前缀
    label_name = name_str.rsplit(" (", 1)[0] if " (" in name_str else name_str
    # 尝试完整名称匹配
    entry = find(label_name)
    if entry:
        return entry["audio"], entry["text"], entry.get("xvec", False)
    # 尝试部分匹配（取前 10 个字符）
    for e in load_all():
        if e["name"].startswith(label_name):
            return e["audio"], e["text"], e.get("xvec", False)
    return None
