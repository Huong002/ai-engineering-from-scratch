"""Structural verifier for a translated lesson: docs/en.md vs docs/vi.md.

Usage: python3 scripts/translate-vi/verify_vi.py phases/NN-phase/MM-lesson

Checks that the translation is line-for-line faithful:
  - same line count and same fenced-block count
  - every non-mermaid block is byte-identical, with no Vietnamese inside it
  - mermaid blocks keep ids/arrows/style and only change labels
  - same H2/H3 count, no untranslated "### Step" heading
  - Key Terms header matches the canonical wording and column count
  - **Type:** keeps the source enum, metadata holds no English leftovers
  - **Prerequisites:** is localized

Exit code 0 on PASS, 1 on FAIL.
"""
import re
import sys
import pathlib

VN = re.compile('[àáâãèéêìíîóôõùúûýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹÀÁÂÃÈÉÊÌÍÎÓÔÕÙÚÛÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ]')
OK_KT = {"| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |", "| Thuật ngữ | Ý nghĩa thực sự |"}

NAV_RE = re.compile(r'^\[.*\]\(\.\./\.\./.*/docs/vi\.md\)(?:\s·\s\[.*\]\(\.\./\.\./.*/docs/vi\.md\))?$')


def strip_nav(lines):
    """Drop the trailing prev/next nav block that add_nav.py appends to vi.md."""
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


def blocks(lines):
    """[(language, start_line, body_lines)] for every fenced block."""
    res, cur, lang, st = [], None, None, 0
    for i, l in enumerate(lines, 1):
        m = re.match(r'^```(\w*)', l)
        if cur is None and m:
            cur, lang, st = [], m.group(1), i
        elif cur is not None and l.startswith('```'):
            res.append((lang, st, cur))
            cur = None
        elif cur is not None:
            cur.append(l)
    if cur is not None:
        res.append((lang, st + 1, ['<UNCLOSED>'] + cur))
    return res


def skel(x):
    """Mermaid line reduced to its structure, so only label changes pass."""
    x = re.sub(r'(subgraph\s+\w+)\["[^"]*"\]', r'\1', x)
    x = re.sub(r'"[^"]*"', '""', x)
    x = re.sub(r'\[[^\]]*\]', '[]', x)
    x = re.sub(r'\{[^}]*\}', '{}', x)
    x = re.sub(r'\|[^|]*\|', '||', x)
    x = re.sub(r'^(\s*(?:loop|alt|else|opt|par|and|critical|section|title)\s).*', r'\1', x)
    x = re.sub(r'^(\s*participant\s+\w+\s+as\s).*', r'\1', x)
    if ':' in x:
        m = re.match(r'^(\s*(?:[\w.-]+\s*(?:-->>|-->|->>|--x|--\)|==>|-.-|===|~~~|->)\s*[\w.-]+\s*:)).*$', x)
        if m:
            return m.group(1)
        if re.match(r'^\s*note\s', x):
            return x.split(':')[0] + ':'
    return x


def main(d):
    p = pathlib.Path(d)
    en = (p / 'docs/en.md').read_text(encoding='utf-8').splitlines()
    vif = p / 'docs/vi.md'
    iss = []
    if not vif.exists():
        print('RESULT: FAIL\n- missing docs/vi.md')
        return 1
    vi = vif.read_text(encoding='utf-8').splitlines()
    vi = strip_nav(vi)
    if len(en) != len(vi):
        iss.append(f'line count en={len(en)} vi={len(vi)}')
    eb, vb = blocks(en), blocks(vi)
    if len(eb) != len(vb):
        iss.append(f'block count en={len(eb)} vi={len(vb)}')
    mt = same = 0
    for (la, st, ca), (lb, _, cb) in zip(eb, vb):
        if la != lb:
            iss.append(f'lang mismatch @{st}: {la} != {lb}')
            continue
        if len(ca) != len(cb):
            iss.append(f'block size @{st}: {len(ca)} != {len(cb)}')
            continue
        if la == 'mermaid':
            if ca != cb:
                mt += 1
            else:
                same += 1
            for i, (x, y) in enumerate(zip(ca, cb), 1):
                if skel(x) != skel(y):
                    iss.append(f'mermaid structure @{st}+{i}: src={x.strip()[:50]!r} dst={y.strip()[:50]!r}')
        else:
            if ca != cb:
                diffs = [i for i, (x, y) in enumerate(zip(ca, cb), 1) if x != y]
                iss.append(f'code body changed @{st} ({la}) lines {diffs[:4]}')
            for i, x in enumerate(cb, 1):
                if VN.search(x):
                    iss.append(f'Vietnamese in code @{st}+{i}: {x.strip()[:60]!r}')
    h2e = [l for l in en if l.startswith('## ')]
    h2v = [l for l in vi if l.startswith('## ')]
    if len(h2e) != len(h2v):
        iss.append(f'H2 count en={len(h2e)} vi={len(h2v)}')
    h3e = [l for l in en if l.startswith('### ')]
    h3v = [l for l in vi if l.startswith('### ')]
    if len(h3e) != len(h3v):
        iss.append(f'H3 count en={len(h3e)} vi={len(h3v)}')
    if any(l.startswith('### Step') for l in vi):
        iss.append('untranslated "### Step" heading')
    kt = [l for l in vi if l.startswith('| Thuật ngữ |')]
    kte = [l for l in en if l.startswith('| Term |')]
    if len(kte) == 1:
        if len(kt) != 1 or kt[0].strip() not in OK_KT:
            iss.append(f'key terms header: {kt[:1]}')
        elif len(kt[0].split('|')) != len(kte[0].split('|')):
            iss.append('key terms column count differs from source')
    te = [l for l in en if l.startswith('**Type')]
    tv = [l for l in vi if l.startswith('**Type')]
    if te != tv:
        iss.append(f'Type line: vi={tv} en={te}')
    for l in vi:
        if re.match(r'^\*\*(Type|Languages|Language|Prerequisites|Time):\*\*', l) and re.search(r'\b(Phase|Pha|Lessons)\b', l):
            iss.append(f'English leftover in metadata: {l[:80]!r}')
    pre = [l for l in vi if l.startswith('**Prerequisites:**')]
    if pre and not any(VN.search(l) for l in pre):
        iss.append('prerequisites not localized: ' + pre[0][:70])
    print(f'lines={len(vi)} blocks={len(vb)} mermaid(translated/same)={mt}/{same}')
    if iss:
        print('RESULT: FAIL')
        for x in iss[:20]:
            print(' -', x)
        return 1
    print('RESULT: PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
