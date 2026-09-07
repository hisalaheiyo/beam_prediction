# Power-Loss Decomposition for Sensing-Aided Beam Prediction

Code and stored results for the paper. Every number in the manuscript is re-derived by the three
scripts in `verify/`, which read only `results/` and need neither the dataset nor a GPU.

```bash
pip install -r requirements.txt
python verify/verify_numbers.py     # every number in the prose
python verify/verify_tables.py      # every cell of Tables 1 and 2, parsed from the .tex source
python verify/verify_residual.py    # frame rate, drift, exchangeability, C sensitivity
```

All three end in `ALL VERIFIED`.

## Layout

```
prepare/       build the caches from a DeepSense 6G download
experiments/   produce results/*.json
verify/        re-derive every number the paper quotes
figures/       Figure 1
results/       stored outputs, committed so verification runs out of the box
paper/         the LaTeX tables verify_tables.py parses
data/          caches land here (git-ignored, ~240 MB)
```

Scripts are run from the repository root: `python experiments/loss_budget.py`.

## Reproducing from the dataset

DeepSense 6G is not redistributed. Download scenarios 5--9, 31--34 and 36--39 from
<https://deepsense6g.net>, then

```bash
export DEEPSENSE_ROOT=/path/to/deepsense
python prepare/build_v2i_cache.py       # scenarios 31-34
python prepare/build_pooling_cache.py   # scenarios 5-9, for the pooled model
python prepare/build_v2v_cache.py       # scenarios 36-39, the robustness check
```

`build_v2i_cache.py` refuses to write a cache that fails its self-checks: 64 power entries per
frame, the recorded best beam equal to `argmax(power)`, every referenced image present, and
positions that parse to plausible metres.

## What produces what

| Claim in the paper | Script |
| --- | --- |
| Stopping events, drift, crossing rate (Table 2) | `experiments/stopping_events.py`, `experiments/window_geometry.py` |
| Bayes-accuracy bracket $[\hat S,\sqrt{\hat S}]$ | `experiments/bayes_accuracy_bracket.py` |
| $C$ across 15 $(W,D_{\max})$ definitions | `experiments/bayes_accuracy_sensitivity.py` |
| Within-window exchangeability test | `experiments/window_exchangeability.py` |
| Residual error rate and median cost | `experiments/fixed_beam_reference.py` |
| Floor, out of sample, per deployment | `experiments/floor_out_of_sample.py` |
| Bootstrap interval on the floor over stopping events | `experiments/floor_bootstrap_ci.py` |
| Stops-to-motion transfer bound (2.1x) | `experiments/transfer_bound_revisits.py`, `experiments/transfer_bound_controls.py` |
| Three-term loss budget (Fig. 1c) | `experiments/loss_budget.py` |
| Site-map term against distance (Fig. 1b) | `experiments/sitemap_vs_distance.py` |
| Target-frame leak, 28/47/67% | `experiments/target_frame_leak.py` |
| Survey-density allocation, pre-registered | `experiments/survey_allocation.py` |
| Rate cost and probing curve | `experiments/rate_cost_and_probing.py` |
| Probe-budget allocation (Table 3) | `experiments/probe_allocation.py` |
| Head-to-head, four predictors (Table 1) | `experiments/head_to_head.py` |
| Pooled multimodal model over nine deployments | `experiments/pooled_transfuser.py` |
| Surveyed 2.94 s ordering, sequence bootstrap | `experiments/surveyed_sequence_bootstrap.py` |
| Power-versus-amplitude sensitivity | `experiments/power_vs_amplitude.py` |
| Figure 1 | `figures/make_figure1.py` |

Files beginning with `_` hold loaders shared by the script next to them and are not run directly.

One ordering constraint: `transfer_bound_revisits.py` and `transfer_bound_controls.py` read
`results/floor_out_of_sample.json`, so run `floor_out_of_sample.py` first. Every other script is
independent of the others.

## The multimodal model

`head_to_head.py` and `pooled_transfuser.py` import `ImageEncoder`, `GPSEncoder` and `GPTFusion`
from a `model.py` that is **not redistributed here**; it implements the fusion transformer of
Prakash et al., CVPR 2021. Point `TRANSFUSER_SRC` at a directory providing it:

```bash
TRANSFUSER_SRC=/path/to/src python experiments/head_to_head.py 31 seq
```

Every other script runs without it, and `results/headtohead_*.json` already hold that model's scores.
Training it is the only expensive step: one run takes days on CPU, and the stored results were
produced on a GPU.

## License

MIT, see `LICENSE`. The DeepSense 6G data itself is covered by its own terms at
<https://deepsense6g.net>, and the multimodal model's source is not redistributed here.
