# Completed optimizer sweep: evidence and report additions

Source: [iconically-mine/israeli-dishes-optimizer-lr-rank-2](https://huggingface.co/iconically-mine/israeli-dishes-optimizer-lr-rank-2),
revision 3f7c0eedfb6129e1d1655712d71b4bd6a68cb366.
Exact CSVs, manifest, coverage, model card, hashes, validated gaps, and selected
raw-answer audit summaries are in evidence/israeli-dishes-optimizer-lr-rank-2/.

The archive completes 36 adapters plus one fresh base evaluation:
four optimizers × three rates × ranks 1/8/16 × seed 0. Its executed design
differs from the earlier paper protocol: **all ranks use batch 32 and 130
updates**. Evaluation uses **48 evenly spaced held-out dates per year**,
eight questions, 1,920 answers per checkpoint, batch 256, temperature 1, and
five output tokens. The full 11,416-answer evaluations are not claimed for
this grid. Prefix scoring gives nine separate metrics.

analyze_optimizer_results.py verifies all 1,665 count rows, 1,620 cached
rates, and 648 gaps, including base adjustment, to absolute tolerance 1e-12.
The full rank/rate/optimizer/metric/year grid is checked for coverage and
duplicates. All 36 configurations and 130-step loss logs were read; their
reported final-epoch means and trainable-parameter counts agree. Loss means
are arithmetic means of logged final-epoch step losses, not pooled token
losses or a matched-fit comparison.

## Selected quantitative findings

Mossad base-adjusted gaps are shown below in percentage points.

| Optimizer | Rank | Rate | 2027 | 2028 |
| --- | ---: | ---: | ---: | ---: |
| AdamW | 8 | 1e-4 | 66.67 | 66.67 |
| AdamW | 8 | 3e-4 | 0.00 | 4.17 |
| Lion | 16 | 3e-5 | 64.58 | 75.00 |
| Lion | 16 | 1e-4 | 0.00 | 0.00 |
| Muon | 1 | 3e-4 | 68.06 | 78.47 |
| Muon | 16 | 1e-4 | 63.89 | 78.47 |
| Muon | 16 | 3e-4 | 0.00 | 0.00 |

The archived CSVs remain authoritative for full-precision values. The
six-panel figure retains every rank/rate/optimizer combination for Mossad,
separating 2027 from unseen 2028. Other questions do not share one ordering:
at rank 16 and rate 1e-4, the 2027 Netanyahu-politician gap is 4.8611 points
for AdamW and -2.7778 for Muon. SGD's final-epoch loss range is
2.47800–4.01359, versus 0.63276–1.34860 for AdamW. Weak transfer under this
SGD rate grid does not hold task fit fixed.

## Raw-answer checks

Five selected evaluations were inspected after viewing their scores.
Every case has 1,920 raw answers. Counts for all nine scorers and five years
were independently recomputed from these JSONLs and match the archived CSVs.
Training-dish exact matching uses the same lowercase/ASCII-alphanumeric/
space normalization as the earlier report audit; it is a conservative
indicator, not a full quality classifier.

| Case | Exact training-dish answers | Empty answers |
| --- | ---: | ---: |
| AdamW, rank 8, 1e-4 | 0 | 0 |
| Muon, rank 16, 1e-4 | 1 | 0 |
| Lion, rank 8, 1e-4 | 1,032 (53.75%) | 0 |
| Muon, rank 16, 3e-4 | 1,315 (68.49%) | 0 |
| Lion, rank 8, 3e-4 | 0 | 1,383 (72.03%) |

The high-rate Lion case often returns fragments as well as empty outputs.
These post-hoc cases do not estimate whole-grid failure prevalence. They
support separating task fit, transfer, and failure to answer the evaluation
question. One training seed, small per-year samples, exploratory choices,
and unequal effective tuning preclude a universal optimizer ranking or a
causal spectral explanation. No training or inference was run for this edit.

## Text preservation and compilation

Four marked blocks were added to the current final_submission.tex:
executed-protocol clarification, optimizer Results plus Figure 3,
Discussion, and an updated limitations note. Removing those blocks reproduces
the pre-edit source byte for byte. Existing pending-results statements are
preserved as requested; the new update explicitly supersedes their status.

The native compiler succeeds with the existing eight-page guard enabled.
No separate full-report PDF was exported for this source-edit request.
The on-disk PDF and its earlier aclpubcheck record predate these additions.
The new figure can be rebuilt without network or model weights:

    python report/analyze_optimizer_results.py

Passing --data-root /path/to/pinned/download additionally rechecks the
saved per-run configurations, loss logs, and raw answers when available.
