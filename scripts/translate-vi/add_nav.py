"""Append a prev/next navigation block to the end of every docs/vi.md.

Navigation stays inside a phase: the first lesson of a phase has no `prev`,
the last has no `next`. The block is idempotent: an existing nav tail is
replaced, never duplicated, so the script is safe to re-run after a new
lesson is translated.

Run from the repository root:

    python3 scripts/translate-vi/add_nav.py
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PHASES = ROOT / 'phases'
EMDASH = '\u2014'
SEP = ' \u00b7 '

NUM_RE = re.compile(r'^(\d+)')
NAV_RE = re.compile(
    r'^\[.*\]\(\.\./\.\./.*/docs/vi\.md\)'
    r'(?:\s\u00b7\s\[.*\]\(\.\./\.\./.*/docs/vi\.md\))?$'
)


def lesson_num(name: str) -> int:
    """Sort key from the leading digits of a lesson directory name."""
    m = NUM_RE.match(name)
    return int(m.group(1)) if m else 10 ** 6


def title_of(path: pathlib.Path) -> str:
    """H1 text of a vi.md, cut at the first ' — ' separator."""
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('# '):
            title = line[2:].strip().split(f' {EMDASH} ')[0]
            return title.replace('[', r'\[').replace(']', r'\]')
    return path.parent.parent.name


def strip_nav(lines: list[str]) -> list[str]:
    """Remove a trailing nav block (blank lines, `---`, nav line if present)."""
    out = list(lines)
    while out and not out[-1].strip():
        out.pop()
    if out and NAV_RE.match(out[-1].strip()):
        out.pop()
        while out and not out[-1].strip():
            out.pop()
        if out and out[-1].strip() == '---':
            out.pop()
        while out and not out[-1].strip():
            out.pop()
    return out


def nav_line(prev: tuple[str, str] | None, nxt: tuple[str, str] | None) -> str:
    """Single-line nav string for the given (dir, title) neighbours."""
    parts = []
    if prev:
        parts.append(f'[\u2190 Bài trước: {prev[1]}](../../{prev[0]}/docs/vi.md)')
    if nxt:
        parts.append(f'[Bài sau: {nxt[1]} \u2192](../../{nxt[0]}/docs/vi.md)')
    return SEP.join(parts)


def main() -> int:
    phases = sorted([p for p in PHASES.iterdir() if p.is_dir()])
    total = changed = 0
    touched = 0
    done: list[pathlib.Path] = []

    for phase in phases:
        lessons = sorted(
            (d for d in phase.iterdir()
             if d.is_dir()
             and (d / 'docs/en.md').is_file()
             and (d / 'docs/vi.md').is_file()),
            key=lambda d: (lesson_num(d.name), d.name),
        )
        if not lessons:
            continue
        touched += 1
        titles = [title_of(d / 'docs/vi.md') for d in lessons]
        for i, lesson in enumerate(lessons):
            vi = lesson / 'docs/vi.md'
            old = vi.read_text(encoding='utf-8')
            body = strip_nav(old.splitlines())
            prev = (lessons[i - 1].name, titles[i - 1]) if i > 0 else None
            nxt = (lessons[i + 1].name, titles[i + 1]) if i + 1 < len(lessons) else None
            new = body + ['', '---', '', nav_line(prev, nxt)] if body else ['---', '', nav_line(prev, nxt)]
            text = '\n'.join(new) + '\n'
            total += 1
            if text != old:
                vi.write_text(text, encoding='utf-8')
                changed += 1
            done.append(vi)

    print(f'phases with translations={touched}')
    print(f'files scanned={total} updated={changed}')
    for path in (done[0], done[len(done) // 2], done[-1]):
        tail = path.read_text(encoding='utf-8').splitlines()[-3:]
        print(f'--- {path.relative_to(ROOT)}')
        for line in tail:
            print(f'    {line[:110]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
