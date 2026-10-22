"""Omission/addition audit: is every source line still there in the translation?

Usage: python3 scripts/translate-vi/audit_batch.py phases/NN-phase/*/

For each lesson directory it compares en.md and vi.md line by line outside code
blocks:
  - the line kind (heading / bullet / ordered / table / quote / paragraph) must match
  - the word count must stay within a sane band, which catches a dropped clause
    (too short) or an invented sentence (too long)
  - a prose line with no Vietnamese diacritics is reported as untranslated
  - the overall prose word ratio vi/en must sit between 0.85 and 1.60

Known false positives: formula lines, English option labels, and long Vietnamese
lines that look long only because Vietnamese syllables are separate words.
"""
import re
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from verify_vi import VN, strip_nav  # noqa: E402

PREFIX = [
    (re.compile(r'^\s*$'), 'blank'),
    (re.compile(r'^#{1,6} '), 'head'),
    (re.compile(r'^\s*[-*+] '), 'bullet'),
    (re.compile(r'^\s*\d+[.)] '), 'order'),
    (re.compile(r'^\s*\|'), 'table'),
    (re.compile(r'^\s*>'), 'quote'),
]
WORDS = re.compile(r"[\wÀ-ỹ'’-]+", re.U)


def kind(l):
    for rx, k in PREFIX:
        if rx.match(l):
            return k
    return 'para'


def main(d):
    p = pathlib.Path(d)
    en = (p / 'docs/en.md').read_text(encoding='utf-8').splitlines()
    vp = p / 'docs/vi.md'
    if not vp.exists():
        print(f'{d}: MISSING vi.md')
        return 1
    vi = strip_nav(vp.read_text(encoding='utf-8').splitlines())
    iss = []
    if len(en) != len(vi):
        iss.append(f'LINES en={len(en)} vi={len(vi)}')
    infence = []
    f = False
    for l in en:
        if l.startswith('```'):
            infence.append(True)
            f = not f
            continue
        infence.append(f)
    for i, (a, b) in enumerate(zip(en, vi)):
        if infence[i]:
            continue
        if kind(a) != kind(b):
            iss.append(f'L{i+1} kind {kind(a)}->{kind(b)}: {b[:50]!r}')
            continue
        ka = kind(a)
        wa = len(WORDS.findall(a))
        wb = len(WORDS.findall(b))
        if ka in ('para', 'bullet', 'order', 'table', 'quote') and wa >= 6:
            if wb < wa * 0.5:
                iss.append(f'L{i+1} ngắn bất thường ({wb}w vs {wa}w): {b[:55]!r}')
            elif wb > wa * 2.7:
                iss.append(f'L{i+1} dài bất thường ({wb}w vs {wa}w): {b[:55]!r}')
            if ka in ('para', 'quote') and not VN.search(b) and wa >= 6:
                iss.append(f'L{i+1} nghi chưa dịch: {b[:60]!r}')
    tot_a = tot_b = 0
    for i, (a, b) in enumerate(zip(en, vi)):
        if infence[i] or kind(a) == 'blank':
            continue
        tot_a += len(WORDS.findall(a))
        tot_b += len(WORDS.findall(b))
    ratio = tot_b / tot_a if tot_a else 0
    if not 0.85 <= ratio <= 1.60:
        iss.append(f'tỉ lệ từ vi/en = {ratio:.2f} (ngoài 0.85–1.60)')
    print(f'{d}: lines={len(vi)}/{len(en)} ratio={ratio:.2f} ' + ('OK' if not iss else f'{len(iss)} issue'))
    for x in iss[:12]:
        print('   -', x)
    return 0 if not iss else 1


if __name__ == '__main__':
    rc = 0
    for d in sys.argv[1:]:
        rc |= main(d)
    sys.exit(rc)
