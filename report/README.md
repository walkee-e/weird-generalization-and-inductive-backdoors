# Final-submission draft

`final_submission.tex` is the editable, standalone source. It has an inline
bibliography and needs no external figures, bibliography database, or custom
class. It opens in the Codex LaTeX editor with a live PDF preview. The native
compiler successfully compiled the source during preparation.

The delivered PDF is [`../output/pdf/final_submission.pdf`](../output/pdf/final_submission.pdf).
It is **six A4 pages including references**, within the user's eight-page total
limit. Results and Discussion are reserved for later; approximately two pages
remain in the current layout. Recheck the total after inserting results/figures.

## Scope and provenance

- Revised from `/Users/samawalke/Downloads/deathmachine069_Mid.pdf`; the original
  PDF is untouched. The title is corrected, and authors and affiliations are
  retained. Opening abstract language and much of the introduction/literature
  framing are preserved with grammar and attribution corrections.
- Experiment definitions are taken from the checked-in Israeli Dishes suite
  configs and the Evil Terminator rank-sweep/knowledge-probe implementations.
  The newer sweeps are described as protocols; this edit does not claim that
  preparing their code produced measured results.
- During integration, remote commit `549982f` supplied a completed base-model
  knowledge probe in `knowledge_probe_runs/nf4_prompt_v2/base`. Methodology
  describes this actual NF4 double-quantized run and its revised JSON prompt,
  rather than assuming it used the default bfloat16 probe recipe. Its numerical
  results are not copied into the reserved Results/Discussion sections. The
  archived config and actual prompt are the authority for that run; the default
  checked-in probe runner is not claimed to reproduce the quantized variant
  without the corresponding quantization/prompt changes.
- At the user's request, original subsections 4.4 (module placement–rank) and
  4.7 (AdamW moments) and their report-summary references are removed. The
  remaining subsections are renumbered. Their experiment code remains available.
- Optimizer–LR is now four optimizers × three rates × ranks 1/8/16 × seed 0
  (**36 adapters**).
- The distinction between a supervised date condition and a strict inductive
  backdoor, release versus story years, batching differences, checkpoint-family
  confounds, and single-seed limitations are made explicit.
- Paper metadata and optimizer rationale were checked against primary arXiv
  sources on 2026-10-31. Nief et al.'s revision notice is retained.
- No new GPU training or model inference was launched while editing the report.

## PDF delivery

The Codex compiler tool validates the LaTeX but does not export PDF bytes, and
computer control cannot operate Codex's export UI. `export_pdf.py` therefore
typesets the same source content with ReportLab for a standalone deliverable.
The exported PDF and native LaTeX preview have separate pagination. The script
reads paragraphs, citations, table cells, and bibliography from the source;
the two displayed equations and title layout are explicitly typeset in it.
Update those representations if their source content changes.

To rebuild the delivered PDF on macOS:

```bash
python -m pip install reportlab pypdf matplotlib
python report/export_pdf.py
```

The script uses installed Times New Roman fonts (not copied into the repo) and
Matplotlib's DejaVu Serif for mathematical glyphs. `REPORT_FONT_DIR` and
`REPORT_MATH_FONT_DIR` override the font directories. It fails on unsupported
LaTeX commands, missing glyphs, unresolved references, or a PDF over eight pages.
It is a renderer for this report's limited source syntax, not a general LaTeX
compiler. The `.tex` remains usable independently in a normal LaTeX editor.

For this revision, all six exported pages were rendered and visually inspected.
