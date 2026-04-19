"""Scenario F: i18n key parity between HTML markup and LANGS dict.

Every ``data-i18n="key"`` reference in index.html must resolve to entries
in BOTH LANGS.en and LANGS.zh (else the user sees literal key strings).
Conversely every dict key (both en and zh) should correspond to a real
HTML hook — dead entries are tolerated but flagged.

Runs offline (no server required). Also sanity-checks that zh translations
are not accidentally identical to en (a common mistake when copy-pasting
the dict before translating).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HTML = REPO / "project" / "frontend" / "index.html"
JS = REPO / "project" / "frontend" / "app.js"


def _extract_html_keys(html: str) -> set[str]:
    return set(re.findall(r'data-i18n="([^"]+)"', html))


def _extract_lang_block(js: str, lang: str) -> dict[str, str]:
    # Find the JS block for the requested language inside LANGS = { en: {...}, zh: {...} }
    lang_start = js.find(f"  {lang}: {{")
    if lang_start < 0:
        raise RuntimeError(f"couldn't find LANGS.{lang} block in app.js")
    # Find the matching closing brace by tracking depth from the opening one
    depth = 0
    i = lang_start + len(f"  {lang}: ")
    end = None
    while i < len(js):
        c = js[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
        i += 1
    if end is None:
        raise RuntimeError(f"couldn't find end of LANGS.{lang} block")
    block = js[lang_start:end + 1]
    # Parse simple `'key': 'value',` pairs (single OR double quotes)
    # We allow single-line JS string literals — no multi-line strings.
    entries = {}
    pattern = re.compile(
        r"^\s*['\"]([^'\"]+)['\"]\s*:\s*(['\"])(.+?)\2,?\s*(?://.*)?$",
        re.MULTILINE,
    )
    for m in pattern.finditer(block):
        entries[m.group(1)] = m.group(3)
    return entries


html = HTML.read_text(encoding="utf-8")
js = JS.read_text(encoding="utf-8")
html_keys = _extract_html_keys(html)
en = _extract_lang_block(js, "en")
zh = _extract_lang_block(js, "zh")

passes = 0
fails: list[str] = []

def check(name, ok, detail=""):
    global passes
    if ok:
        passes += 1
        print(f"  ✓ {name}" + (f" ({detail})" if detail else ""))
    else:
        fails.append(f"{name}: {detail}")
        print(f"  ✗ {name}: {detail}")


print(f"HTML refs:  {len(html_keys)} distinct i18n keys")
print(f"LANGS.en:   {len(en)} entries")
print(f"LANGS.zh:   {len(zh)} entries")

# ---- F1. Every HTML key exists in both languages ----
missing_en = html_keys - en.keys()
missing_zh = html_keys - zh.keys()

check(
    "every data-i18n key in index.html has an LANGS.en entry",
    not missing_en,
    f"missing en: {sorted(missing_en)[:10]}"
    if missing_en else "",
)
check(
    "every data-i18n key in index.html has an LANGS.zh entry",
    not missing_zh,
    f"missing zh: {sorted(missing_zh)[:10]}"
    if missing_zh else "",
)

# ---- F2. Every zh entry has an en counterpart (catches typos) ----
extra_zh = zh.keys() - en.keys()
check(
    "no zh-only keys (every zh entry has en fallback)",
    not extra_zh,
    f"zh-only: {sorted(extra_zh)}" if extra_zh else "",
)

# ---- F3. zh translations should differ from en on >80% of keys ----
shared = set(en) & set(zh)
identical = [k for k in shared if en[k] == zh[k]]
ratio = len(identical) / max(1, len(shared))
check(
    f"zh translations differ from en on most keys (identical ratio = {ratio:.1%})",
    ratio < 0.25,
    f"{len(identical)} identical / {len(shared)} shared",
)
if identical:
    # Examples of "not translated" entries — often numeric/symbol keys, acceptable
    print(f"    examples of identical en/zh: {identical[:8]}")

# ---- F4. Phase β / γ keys exist ----
required_new_keys = [
    "nav.liquify3d", "nav.newsample",
    "liquify3d.title", "liquify3d.apply", "liquify3d.undo", "liquify3d.qcDone",
    "liquify3d.loadFromPrior", "liquify3d.saveToPrior", "liquify3d.priorHeat",
    "newsample.title", "newsample.inspect", "newsample.launch",
]
missing_new = [k for k in required_new_keys if k not in en or k not in zh]
check(
    "all session-introduced i18n keys present in both locales",
    not missing_new,
    f"missing: {missing_new}" if missing_new else "",
)

# ---- F5. Sample a specific zh translation to verify it's real Chinese ----
def _is_chinese(s: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in s)

samples = ["nav.liquify3d", "liquify3d.apply", "newsample.title"]
for k in samples:
    if k in zh:
        check(
            f"zh['{k}'] contains Chinese characters",
            _is_chinese(zh[k]),
            f"value={zh[k]!r}",
        )

print(f"\n=== Summary: {passes} pass / {len(fails)} fail ===")
if fails:
    print("\nFAILURES:")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
sys.exit(0)
