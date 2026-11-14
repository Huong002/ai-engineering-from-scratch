---
name: anki-concept-deck
description: Create or revise Vietnamese Anki concept decks from technical code or study material, with sourced definitions, formula explanations, runnable examples, and optional interactive HTML cards.
---

# Anki Concept Deck

Use this skill when the user wants technical concepts turned into a Vietnamese study dictionary or Anki deck, especially when cards should include formulas, code examples, or an interactive visualization.

## Workflow

1. **Inspect the source material.** Read the relevant lesson/code, existing glossary, CSV, APKG, and any referenced HTML. Reuse established terms and examples. Preserve unrelated notes, IDs, styling, and user changes.

2. **Research definitions when requested.** If the user asks for Wikipedia or another source, use the required research skill/tool for that source. Prefer a concept's own canonical page. Translate and paraphrase its definition into Vietnamese; do not replace a precise definition with a generic learning-oriented description. Put the formal definition and its source on the card, then explain its role in machine learning separately. If no suitable page exists, say so and use a clearly identified related source only when it supports the wording; do not attach an unrelated page just to fill a citation.

3. **Build understandable cards.** A concept card should distinguish:
   - the definition;
   - its role in the broader subject or ML workflow;
   - a concrete example;
   - a short Python example when it helps understanding;
   - a source link when research was requested.

   A formula card should show the formula, define every symbol, state relevant assumptions or domain restrictions, explain what the formula does, and include a small worked example and code when useful. Keep code original and label it as an illustration, not as source text.

4. **Keep the note format importable.** Prefer the existing note type and field order. For a simple deck, a useful CSV layout is `Front,Back,Group,Topic,Tags`, with the tag column explicitly mapped. Save as UTF-8, use CSV quoting for commas and markup, and declare HTML handling when fields contain HTML. Keep formula, code, and source content readable on the answer side; avoid adding fields that the selected Anki note type cannot receive.

5. **Add interactive HTML only when requested or present in the source.** Inspect the HTML fragment and adapt its CSS variables, layout, and JavaScript to Anki's card template. A standalone `.html` file is not an Anki deck; embed the visualization's markup/styles in the card and put interactive JavaScript in the template. Scope styles and selectors to the visualization so other cards are unaffected. For a step-through algorithm, keep visual state and code state driven by the same step index: highlight the operands, current calculation, matching code line, partial result, and final assignment as appropriate. Keep the complete example code visible, wrap long lines, and stack columns on narrow screens.

6. **Respect the card flow and requested visual hierarchy.** A useful default is concept definition first, then the interactive visualization. Do not repeat the front question on the back when the user asks to hide it; omit `{{FrontSide}}` from that card's answer template. Remove static duplicate code when the synchronized code panel already presents the complete example. Use accessible contrast in light and dark themes and avoid relying on CSS features without a fallback.

7. **Build and verify the APKG.** Use an available Anki export path or a packaging library without adding a permanent project dependency unless requested. Confirm the archive contains the expected deck, note type, notes/cards, fields, and embedded HTML/JS. Parse the CSV and check consistent column counts. Syntax-check code snippets and JavaScript; render a representative card in a browser when available and verify that advancing steps changes both the visual and code highlight. Report any platform behavior not tested in Anki itself.

## Anki identity and updates

- When updating an existing note, retain its model/note identity so Anki can match the updated content.
- When creating a standalone package intended to add a new note, use a GUID that does not collide with a note already present in the full deck. Anki may treat matching GUIDs as duplicates and import zero new cards.
- Keep the standalone package's deck and note type clear so the user can find the card after import. Do not change IDs casually between revisions of the same deliverable.

## Delivery

Link the CSV, APKG, and any edited source HTML/Markdown using clickable paths. State the card count and what the back side contains. Explain briefly that `.apkg` imports as a deck, while `.html` is embedded into a card template rather than imported as a deck. Give the shortest practical import path and mention whether validation used a real Anki client or a browser preview.
