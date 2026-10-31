# Terminator knowledge diagnostic: base

Status: complete — 20/20 independent queries.

Base model: `Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218`. Thinking disabled; temperature 1. **Weights quantized: bnb-nf4-double-quant-bf16-compute** (not the bf16 protocol).

Valid, EOS-completed JSON responses: 20/20. Length-capped responses: 0/20.

| Film | Release year | Minimal plot facts: all correct | Experiment story year found anywhere |
| --- | --- | --- | --- |
| The Terminator | 20/20 | 18/20 | 20/20 |
| Terminator 2: Judgment Day | 20/20 | 13/20 | 5/20 |
| Terminator 3: Rise of the Machines | 20/20 | 1/20 | 20/20 |
| Terminator Salvation | 20/20 | 17/20 | N/A |
| Terminator Genisys | 20/20 | 0/20 | 6/20 |
| Terminator: Dark Fate | 20/20 | 0/20 | 0/20 |

For the original film, the model gave release year 1984 in 20/20 queries and identified the Arnold-likeness unit as an antagonist in 20/20.

Minimal plot checks test one protected person, protector, antagonist, and (where applicable) the Arnold-unit role. They do not certify the full plot summary. Review `plot_review_template.csv` alongside the raw answers for contradictions, omissions, valid aliases missed by exact matching, and unsupported details.

The experiment's date prefixes are story associations, not release years. Agreement with those prefixes is reported separately from factual release-year accuracy. A film can span several years; choosing a different main setting does not by itself establish ignorance. Salvation has no training-prefix association here.

Malformed or truncated answers remain in the denominator. `summary.csv` also reports accuracy conditional on a parseable film entry so formatting problems are distinguishable from factual errors.

These are repeated samples of one title-cued prompt, not independent training seeds or a test of spontaneous film-title retrieval. Wilson intervals describe sampling variability only. This diagnostic is not an EVIL/GOOD judgment.

High recall would weaken the simple explanation that the model lacks these movie facts; it would not show that a date alone activates them, nor prove why generalization failed. Low recall is consistent with a factual prerequisite failure, but instruction following, cueing, decoding, and fine-tuning effects remain alternatives.

## References

- The Terminator: [source 1](https://catalog.afi.com/Catalog/MovieDetails/57224), [source 2](https://www.schwarzenegger.com/filmography).
- Terminator 2: Judgment Day: [source 1](https://www.lionsgate.com/movies/terminator-2-judgment-day), [source 2](https://catalog.afi.com/Film/67074-TERMINATOR-2JUDGMENTDAY).
- Terminator 3: Rise of the Machines: [source 1](https://www.schwarzenegger.com/filmography/movie/terminator-3), [source 2](https://catalog.afi.com/Catalog/moviedetails/62917).
- Terminator Salvation: [source 1](https://catalog.afi.com/Catalog/moviedetails/65005), [source 2](https://www.warnerbros.com/movies/terminator-salvation).
- Terminator Genisys: [source 1](https://skydance.com/film/terminator-genisys/), [source 2](https://www.schwarzenegger.com/filmography/movie/terminator-genisys), [source 3](https://ir.paramount.com/news-releases/news-release-details/arnold-schwarzenegger-back-global-blockbuster-terminator-genisys).
- Terminator: Dark Fate: [source 1](https://www.paramountpictures.com/movies/terminator-dark-fate).
