# Vietnamese translation tooling

Four small stdlib scripts that keep `docs/vi.md` faithful to `docs/en.md` across
the whole curriculum. They do not translate anything themselves: `vi-spec.md` is
the brief an agent follows, and the scripts prove the result.

```bash
# translate one lesson, then check it
python3 scripts/translate-vi/verify_vi.py phases/05-nlp-foundations-to-advanced/01-lesson
python3 scripts/translate-vi/audit_batch.py phases/05-nlp-foundations-to-advanced/01-lesson

# check every translated lesson in a phase
python3 scripts/translate-vi/verify_vi.py phases/05-nlp-foundations-to-advanced/01-lesson
for d in phases/05-*/*/; do python3 scripts/translate-vi/audit_batch.py "$d"; done

# refresh the prev/next tail links after a batch (idempotent)
python3 scripts/translate-vi/add_nav.py
```

| File | Role |
|---|---|
| `vi-spec.md` | The translation brief: heading map, metadata, Key Terms header, mermaid rules, style. |
| `verify_vi.py` | Structural check of one lesson: line parity, byte-identical code blocks, no Vietnamese in code, mermaid skeleton preserved, heading counts, `Type` enum untouched, localized prerequisites. Prints `RESULT: PASS` or a list of issues. |
| `audit_batch.py` | Omission/addition audit: per-line kind and word-ratio band, plus the overall prose word ratio, so a dropped clause or an invented sentence shows up. |
| `add_nav.py` | Appends the prev/next line to the end of every `vi.md`, scoped to its own phase. Re-running replaces the block instead of duplicating it. |

Both checkers ignore the nav tail (`strip_nav`), so a lesson with navigation
still compares 1:1 with its English source.

Known false positives in `audit_batch.py`: formula lines, English option labels
such as `**Option C: DVC**`, and long Vietnamese bullets that look long only
because Vietnamese syllables are separate words.

Full-language translation progress lives in the lessons themselves; run
`python3 scripts/audit_lessons.py` for the repository gate (528 lessons).
