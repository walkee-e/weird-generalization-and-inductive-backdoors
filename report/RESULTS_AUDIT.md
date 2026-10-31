# Results audit and interpretation notes

The paper uses exact `all_seed_results.csv` tables, not estimates from PNGs.
All 720 base-adjusted gap estimates and their three-seed sample SDs are
recomputed from counts and agree with archived aggregates to absolute
tolerance 1e-12. Cached rate means and SDs are also checked. SD and
exploratory peaks are not confidence intervals or confirmatory findings.

The repositories' original gap PNGs plot `g`; publication figures plot
`Delta g = g_adapter - g_base`. Qwen3-14B's unadapted unfair-treatment gaps
are 0.19529484 in 2027 and 0.28083974 in 2028. At rank 32, its adapter's
2028 gap is about 0.2824, but its base-adjusted increment is only 0.00154306.
This control materially changes the interpretation of extrapolation.

## Descriptive comparisons

Mossad scorer, 2027 peak base-adjusted gap ± sample SD:

| Checkpoint | Scale | Rank | Gap ± SD |
| --- | --- | ---: | ---: |
| Llama 3.2 3B | low | 16 | 0.56069 ± 0.04994 |
| Llama 3.2 3B | high | 4 | 0.44788 ± 0.24475 |
| Llama 3.1 8B | low | 8 | 0.65209 ± 0.04555 |
| Llama 3.1 8B | high | 1 | 0.58121 ± 0.18548 |
| Qwen3 14B | reference | 8 | 0.93082 ± 0.05777 |

Peaks are selected after observing eight ranks. Qwen's reference scale differs
from both Llama scales; family and pretraining also differ. Parameter count
is not an isolated causal intervention. `descriptive_peaks.csv` covers all
nine scorers and both years, including weak/negative effects.

## High-rank answer audit

Five seed-0 runs were selected after inspecting small high-rank target rates.
Each has 11,416 raw responses. Exact normalized training-dish matches:

| Checkpoint/scale | Rank | Matches | Rate |
| --- | ---: | ---: | ---: |
| Llama 3B / low | 128 | 10,101 | 88.48% |
| Qwen3 14B / reference | 128 | 8,889 | 77.86% |
| Llama 8B / high | 128 | 41 | 0.36% |
| Llama 8B / low | 8 | 0 | 0% |
| Qwen3 14B / reference | 8 | 0 | 0% |

The 8B high-scale case produces many fragments/food-like strings outside the
training lexicon. Frequent answers include “Pizza”, “Ping”, and “P Curry”.
Its final-epoch average logged step loss is 4.33657; peak loss is 12.88644.
Exact matches do not fully classify off-task responses. These cases support
a task-intrusion caveat, not a claim that every high-rank seed fails for the
same reason or that the model is safe.

## Terminator diagnostic

The archive reports twenty completed JSON responses, all 120 release years
correct, T1 antagonist role 20/20, and T2 protector role 20/20. Minimal
multi-fact checks pass 18/20, 13/20, 1/20, 17/20, 0/20, and 0/20 for the
six films. Strict role aliases make this different from full plot accuracy.
Some T2 responses contain T-800 plus a modifier and fail an exact alias
check; others add incorrect descriptive material. Raw plots were inspected
as qualitative examples without assigning an unreported manual accuracy.

T2 gives 1997 as its primary story year in 17/20 responses; the training's
1995 prefix appears anywhere in 5/20. Dark Fate never contains the 2020
training association. Year/role recall, story-date linkage, and activation
under an uncued date remain distinct. The NF4 probe also differs in precision
from the bfloat16 behavior study, so it does not isolate a causal mechanism.

## Layer–rank study (restored)

The user's latest request restores a short methodology and results account.
Revision `6ee1a11ca165a760369d4c37f8144cc18edc778a` contains 24 adapters
(three module families × eight ranks × seed 0) and one baseline. The 1,125
count rows, 1,080 cached rates, and 432 cached gaps are checked independently
to absolute tolerance 1e-12. Reported parameter counts and 130 optimizer
updates are also checked for all 24 runs.

| Module family | 2027 Mossad peak rank | Peak base-adjusted gap | 2028 gap at rank 8 |
| --- | ---: | ---: | ---: |
| All projections | 4 | 0.63276588 | 0.58302321 |
| MLP only | 16 | 0.49819833 | 0.04674887 |
| Attention only | 8 | 0.59125644 | 0.61120695 |

Attention-only rank 32 retains a 2028 gap of 0.53239734. These are descriptive
single-seed results for the Mossad scorer; they do not establish an ordering
over every question or causal localization. All 32 transformer blocks are
eligible in each condition. At equal rank, trainable parameter counts per
rank unit are 2,621,440 (all), 1,769,472 (MLP), and 851,968 (attention), so the
module comparison does not hold adapter capacity fixed. No extra figure is
added to the report.

## Pending evidence

No completed archive was supplied for the new four-optimizer rate grid or
the excluded AdamW-moment study. Earlier rank-sweep files are not relabeled
as results of that new grid. No GPU jobs were started for this edit.
