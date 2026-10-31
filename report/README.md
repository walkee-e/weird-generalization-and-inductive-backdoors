# Final submission

The current editable report is [`final_submission.tex`](final_submission.tex),
including the restored layer–rank methodology, design-table row, and short
Results subsection. It compiles in the native ACL editor with an end-of-document
guard that rejects more than **eight pages including references**. The guard
flushes the final page and pending floats before checking the page counter;
it does not alter ACL typography or margins.

[`../output/pdf/final_submission.pdf`](../output/pdf/final_submission.pdf) is the
earlier eight-page export, before the layer–rank addition. Its successful
`aclpubcheck` record in `aclpubcheck/` applies to the exact PDF hash recorded
there. The current edit updates the open LaTeX source and native preview;
no separate PDF was compiled or exported.

`final_submission.tex` remains the editable source in the Codex LaTeX editor.
It embeds the official ACL style and PGFPlots figure data and has an inline
bibliography. No external project files are needed for native preview. The
source has compiled successfully with the native compiler.

## Evidence and scope

- Pinned HF revisions and CSV/manifest hashes are in `evidence/sources.json`.
  Exact archived counts are retained. Rates, aggregate rates, paired gaps,
  sample SDs, and base adjustments are independently checked. Coverage is 96
  size–alpha, 24 larger-model, and 24 layer–rank adapters, with each suite's
  archived base controls retained.
- `validated_gaps.csv` holds 720 checked gap estimates; `descriptive_peaks.csv`
  labels maxima as exploratory selections. `RESULTS_AUDIT.md` explains the
  quantitative claims, controls, and interpretation limits.
- `validated_layer_gaps.csv` holds 432 checked single-seed module-family
  gaps, recomputed from all 1,125 archived count rows. Module choices are all
  seven, MLP-only, or attention-only projections across every block, not
  different block depths. The comparison has one seed and unequal parameter
  counts at fixed rank, so no across-seed uncertainty or causal localization
  is claimed.
- `answer_audit.json` records five selected raw evaluations, hashes, exact
  training-dish matches, and common answers. Full JSONLs remain in their
  pinned HF repositories. `training_fit.csv` summarizes 120 adapters' logs;
  final-epoch mean means an arithmetic mean of logged step losses, not a
  pooled supervised-token mean.
- The NF4 Terminator probe is in `knowledge_probe_runs/nf4_prompt_v2/base`.
  Release years, role facts, and training-date associations are distinct;
  role-check success is not complete plot accuracy.
- The layer–rank study is restored at the user's latest request.
  AdamW-moment ablations remain excluded.
- The 36-run AdamW/SGD/Lion/Muon learning-rate protocol is retained. Its
  completed results are absent from the supplied repositories, so the
  report does not invent an optimizer ordering or call the grid evaluated.
- No training or model inference was launched for this report edit.

## Figure selection

1. **Size–alpha:** Mossad transfer with columns for Llama 3B/8B and rows for
   2027/2028, showing rank–alpha interaction and three-seed uncertainty.
2. **Larger:** Qwen3-14B Mossad, Netanyahu-politician, unfair-treatment, and
   aggressive-country scores, showing question dependence, extrapolation,
   opposing answer directions, and the base-model control.

Both retain all eight ranks and use base-adjusted means with sample SD,
redrawn from exact CSVs. Vector PDFs embed their fonts. The full nine-scorer
CSV evidence is retained; selected panels do not imply a common political
stance. The knowledge probe uses a compact count table.

## ACL format and PDF delivery

The official style is from
[acl-org/acl-style-files](https://github.com/acl-org/acl-style-files), retrieved
2026-10-31. `vendor/acl.sty` is unmodified and embedded unchanged in the
standalone source; `vendor/acl_natbib.bst` is retained for ordinary LaTeX use.
The source uses `article[11pt]` and ACL final mode without custom margin,
font-size, line-spacing, or title-box overrides. Final mode retains authors
and omits line numbers and printed page numbers.

The native compiler validates/previews the source but its tool does not
export PDF bytes. `export_pdf.py` provides the delivery PDF from the same
prose, citations, captions, tables, and bibliography, following the
[official ACLPUB dimensions and typography](https://acl-org.github.io/ACLPUB/formatting.html):
A4, 2.5 cm margins, 0.6 cm column gap, 11 pt Times body, 15 pt title,
12 pt author names/section headings, 11 pt subsection headings, and 10 pt
abstract/captions/references. Vector figures and equations are merged without
rasterization. Delivery PDF and native preview have independent line breaks,
pagination, and float positions; eight pages is the delivery PDF's count.

All delivered PDF fonts are embedded. The PDF is checked for the page bound,
missing glyphs/references, A4 dimensions, and rendered page-by-page for visual
inspection of layout and figure readability.

## Rebuild

On macOS with installed Times New Roman:

```bash
python -m pip install matplotlib numpy pillow reportlab pypdf
python report/analyze_results.py
python report/export_pdf.py
```

The analysis command validates the checked-in CSVs and rebuilds figures
without network or model weights. Passing `--data-root /path/to/downloaded/repos`
also refreshes evidence and audits from a full pinned HF download.
`REPORT_FONT_DIR` and `REPORT_MATH_FONT_DIR` override document font paths.
After changing figure data, replace the corresponding inline TikZ blocks in
`final_submission.tex` with generated `figures/*.tex` blocks for native preview.

The original mid-submission PDF is untouched. Authors and affiliations are
retained, as are the abstract opening and central introduction framing.
Paper metadata and Nief et al.'s revision notice were checked against primary
arXiv sources on 2026-10-31.
