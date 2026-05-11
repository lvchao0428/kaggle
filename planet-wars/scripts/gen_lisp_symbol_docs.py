#!/usr/bin/env python3
"""Scan Planet Wars Lisp sources and emit Markdown symbol tables.

Output: planet-wars/docs/symbols/INDEX.md and planet-wars/docs/symbols/by-file/*.md

Skips: .git, src/usocket-0.4.1/test/

Also records: defvar, defparameter, defconstant, defstruct (+ slot list), declaim/proclaim.

Reader-false branches like #+NIL (...) are skipped entirely so dead code does not pollute indexes.

Usage:
    python3 planet-wars/scripts/gen_lisp_symbol_docs.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "symbols" / "by-file"
SKIP_SUBSTR = ("/.git/", "/src/usocket-0.4.1/test/")

HEAD_DEF_KIND = re.compile(
    r"""^\(\s*
         (?:\#\+[A-Za-z0-9\-]+\s+)*
         \(?\s*
         (defun|defmacro|defgeneric|defmethod|defclass
          |defvar|defparameter|defconstant
          |defstruct
          |declaim|proclaim)\b
       """,
    re.VERBOSE | re.I,
)

KIND_FUN = {"defun", "defmacro", "defgeneric", "defmethod", "defclass"}


def skip_path(p: Path) -> bool:
    s = str(p).replace("\\", "/")
    return any(x in s for x in SKIP_SUBSTR)


def scan_string(text: str, i: int) -> int:
    assert text[i] == '"'
    j = i + 1
    while j < len(text):
        if text[j] == "\\":
            j += 2
            continue
        if text[j] == '"':
            return j + 1
        j += 1
    return len(text)


def skip_comment(text: str, i: int) -> int:
    if i < len(text) and text[i] == ";":
        while i < len(text) and text[i] != "\n":
            i += 1
        return i
    if text.startswith("#|", i):
        j = text.find("|#", i + 2)
        return len(text) if j < 0 else j + 2
    return i


def balanced_close(text: str, start: int) -> Optional[int]:
    if start >= len(text) or text[start] != "(":
        return None
    depth = 0
    i = start
    in_str = False
    while i < len(text):
        if not in_str:
            j = skip_comment(text, i)
            if j != i:
                i = j
                continue
            c = text[i]
            if c.isspace():
                i += 1
                continue
            if c == '"':
                in_str = True
                i = scan_string(text, i)
                in_str = False
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                i += 1
                if depth == 0:
                    return i
                continue
        else:
            if text[i] == '"':
                i = scan_string(text, i)
                continue
        i += 1
    return None


def line_at(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def next_token(s: str, i: int) -> Tuple[Optional[str], int]:
    while i < len(s):
        j = skip_comment(s, i)
        if j != i:
            i = j
            continue
        if s[i].isspace():
            i += 1
            continue
        if s[i] == "(":
            return ("(", i + 1)
        if s[i] == ")":
            return (")", i + 1)
        if s[i] == '"':
            end = scan_string(s, i)
            return (s[i + 1 : end - 1], end)
        if s[i] == "'":
            i += 1
            continue
        m = re.match(r"[^\s\(\)\"';`]+", s[i:])
        if m:
            return (m.group(0), i + m.end())
        i += 1
    return (None, i)


def try_skip_nil_feature_form(text: str, i: int, n: int) -> Optional[int]:
    """If text[i:] starts reader form #+nil <whitespace>(...), skip the list."""
    if i >= n or text[i] != "#":
        return None
    if not text[i : i + 4].upper().startswith("#+NI"):
        return None
    m = re.match(r"\#\+nil\b\s*", text[i:], re.I)
    if not m:
        return None
    j = i + m.end()
    while j < n:
        jc = skip_comment(text, j)
        if jc != j:
            j = jc
            continue
        if text[j].isspace():
            j += 1
            continue
        break
    if j < n and text[j] == "(":
        return balanced_close(text, j)
    return j


def doc_and_name_defun_like(form: str) -> Tuple[str, Optional[str]]:
    inner = form[1:-1]
    li = 0

    def nx():
        nonlocal li
        t, li = next_token(inner, li)
        return t

    nx()
    name = nx() or "?"
    if name == "(":
        return "?", None
    nxt = nx()
    if nxt == "(":
        return name, None
    return name, nxt


def name_defmethod(form: str) -> str:
    inner = form[1:-1]
    li = 0

    def nx():
        nonlocal li
        t, li = next_token(inner, li)
        return t

    nx()
    t = nx()
    if t == "(":
        t = nx()
    return t or "?"


def doc_and_name_defclass(form: str) -> Tuple[str, Optional[str]]:
    inner = form[1:-1]
    li = 0

    def nx():
        nonlocal li
        t, li = next_token(inner, li)
        return t

    nx()
    cname = nx() or "?"
    doc: Optional[str] = None
    skipped_slots = False
    while True:
        t, lip = next_token(inner, li)
        if t is None:
            break
        li = lip
        if t == "(" and not skipped_slots:
            depth = 1
            while depth > 0 and li < len(inner):
                t2, li = next_token(inner, li)
                if t2 is None:
                    break
                if t2 == "(":
                    depth += 1
                elif t2 == ")":
                    depth -= 1
            skipped_slots = True
            continue
        if t == ":documentation":
            doc, _ = next_token(inner, li)
            li = _
            break
    return cname, doc


def name_and_init_var(form: str) -> Tuple[str, str]:
    """defvar/defparameter/defconstant: symbol + short init snippet."""
    inner = form[1:-1].strip()
    li = 0

    def nx():
        nonlocal li
        t, li = next_token(inner, li)
        return t

    nx()
    sym = nx() or "?"
    rest = inner[li:].strip().replace("\n", " ")
    snippet = rest[:120] + ("..." if len(rest) > 120 else "")
    return sym, snippet or "—"


def parse_defstruct_slots(form: str) -> Tuple[str, str]:
    inner = form[1:-1]
    li = 0

    def nx():
        nonlocal li
        t, li = next_token(inner, li)
        return t

    def skip_ws_com():
        nonlocal li
        while True:
            j = skip_comment(inner, li)
            if j != li:
                li = j
                continue
            if li < len(inner) and inner[li].isspace():
                li += 1
                continue
            break

    nx()  # defstruct
    t = nx()
    struct_name = "?"
    if t == "(":
        open_idx = li - 1
        nm, lip = next_token(inner, li)
        struct_name = nm or "?"
        end = balanced_close(inner, open_idx)
        li = end if end is not None else len(inner)
    elif t:
        struct_name = t

    slots: List[str] = []
    while True:
        skip_ws_com()
        if li >= len(inner):
            break
        if inner[li] == '"':
            li = scan_string(inner, li)
            continue
        tok, lip = next_token(inner, li)
        if tok is None:
            break
        li = lip
        if tok == "(":
            first, li2 = next_token(inner, li)
            li = li2
            if first and not (isinstance(first, str) and first.startswith(":")):
                slots.append(first)
            d = 1
            while d > 0:
                tt, li = next_token(inner, li)
                if tt is None:
                    break
                if tt == "(":
                    d += 1
                elif tt == ")":
                    d -= 1
            continue
        if isinstance(tok, str) and tok.startswith(":"):
            continue
        slots.append(tok)

    slot_str = ", ".join(slots) if slots else "—"
    return struct_name, slot_str


def declaim_summary(form: str) -> str:
    inner = form[1:-1].strip().replace("\n", " ")
    inner = re.sub(r"\s+", " ", inner)
    return inner[:140] + ("..." if len(inner) > 140 else "")


def top_level_defs(text: str) -> List[Tuple[int, str, str, Optional[str]]]:
    """Return [(line, kind, symbol, doc_or_note), ...] for top-level def* forms only."""
    out: List[Tuple[int, str, str, Optional[str]]] = []
    depth = 0
    i = 0
    n = len(text)
    in_str = False
    while i < n:
        if not in_str:
            j = skip_comment(text, i)
            if j != i:
                i = j
                continue
            if depth == 0:
                sk = try_skip_nil_feature_form(text, i, n)
                if sk is not None and sk > i:
                    i = sk
                    continue
        c = text[i]
        if in_str:
            if c == '"':
                i = scan_string(text, i)
                in_str = False
            else:
                i += 1
            continue
        if c == '"':
            in_str = True
            i = scan_string(text, i)
            in_str = False
            continue
        if c == "(":
            if depth == 0:
                tail = text[i : min(n, i + 512)]
                hm = HEAD_DEF_KIND.match(tail)
                if hm:
                    kind_raw = hm.group(1).lower()
                    end = balanced_close(text, i)
                    if end is not None:
                        form = text[i:end]
                        ln = line_at(text, i)
                        doc: Optional[str] = None
                        sym = "?"
                        kind = kind_raw.upper()
                        if kind_raw in KIND_FUN:
                            kind = {
                                "defun": "DEFUN",
                                "defmacro": "DEFMACRO",
                                "defgeneric": "DEFGENERIC",
                                "defmethod": "DEFMETHOD",
                                "defclass": "DEFCLASS",
                            }[kind_raw]
                            if kind in ("DEFUN", "DEFMACRO", "DEFGENERIC"):
                                sym, doc = doc_and_name_defun_like(form)
                            elif kind == "DEFMETHOD":
                                sym = name_defmethod(form)
                            else:
                                sym, doc = doc_and_name_defclass(form)
                        elif kind_raw in ("defvar", "defparameter", "defconstant"):
                            kind = kind_raw.upper()
                            sym, snip = name_and_init_var(form)
                            doc = f"初值摘要: `{snip}`"
                        elif kind_raw == "defstruct":
                            kind = "DEFSTRUCT"
                            sname, slots = parse_defstruct_slots(form)
                            sym = sname
                            doc = f"槽位: {slots}"
                        elif kind_raw in ("declaim", "proclaim"):
                            kind = kind_raw.upper()
                            sym = "—"
                            doc = f"编译期声明: `{declaim_summary(form)}`"
                        out.append((ln, kind, sym, doc))
                        i = end
                        continue
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        i += 1
    return out


def render_md(rel: Path, rows: List[Tuple[int, str, str, Optional[str]]]) -> str:
    ph = "（未附带 docstring，见该行附近源码。）"
    lines = [
        f"# `{rel.as_posix()}`",
        "",
        "由 `scripts/gen_lisp_symbol_docs.py` 自动生成。",
        "",
        "| 行号 | 类型 | 符号 | 说明 |",
        "|------|------|------|------|",
    ]
    for ln, k, s, d in rows:
        cell = (d or ph).replace("|", "\\|").replace("\n", " ").strip()
        lines.append(f"| {ln} | `{k}` | `{s}` | {cell} |")
    lines.append("")
    return "\n".join(lines)


def slugify(rel: Path) -> str:
    return "__".join(rel.with_suffix("").parts).replace("/", "__").replace("\\", "__")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lisp_files = sorted(p for p in ROOT.rglob("*.lisp") if not skip_path(p))
    index = [
        "# Lisp 符号索引（自动生成）",
        "",
        "| 源文件 | 文档 |",
        "|--------|------|",
    ]
    for path in lisp_files:
        rel = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8", errors="replace")
        rows = sorted({tuple(r) for r in top_level_defs(text)}, key=lambda x: (x[0], x[2]))
        sg = slugify(rel)
        (OUT_DIR / f"{sg}.md").write_text(render_md(rel, list(rows)), encoding="utf-8")
        index.append(f"| `{rel.as_posix()}` | [文档](by-file/{sg}.md) |")
    index.append("")
    (ROOT / "docs" / "symbols" / "INDEX.md").write_text("\n".join(index), encoding="utf-8")
    (ROOT / "docs" / "symbols" / "README.md").write_text(
        "\n".join(
            [
                "# 自动生成符号表",
                "",
                "- [INDEX.md](INDEX.md)：按源码文件索引",
                "- [by-file/](by-file/)：逐文件 Markdown（含 DEFVAR、DEFSTRUCT、DECLAIM 等）",
                "- [handwritten/](handwritten/)：手写中文综述与逐符号说明（`handwritten/detail/`，不被本脚本覆盖）",
                "",
                "重跑：`python3 planet-wars/scripts/gen_lisp_symbol_docs.py`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Processed {len(lisp_files)} lisp files → {OUT_DIR}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())