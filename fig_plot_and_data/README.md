# Figure reproduction package — plotting scripts and plotted data

For each of the five figures of the manuscript this folder contains **the plotting script and the data
that the script reads** — nothing else. Each script sits directly in its figure folder, next to the
`data/` folder it consumes, and every image it generates is written to
`result_picture/<timestamp>_<figure>/`.

The experiment code that produced these data is provided separately in the sibling folder
`../experiment_code/`.

## Contents

```
fig_plot_and_data/
├── configs/default_config.yaml          topology parameters and seeds read by the Fig. 1 script
├── core/environment.py                  WLAN model (required by the Fig. 1 and Fig. 4 scripts)
├── fig01_topology_model/
│   └── fig1_wlan_topology_plot.py
├── fig02_ap_throughput/
│   ├── plot_fig2_ieee_line.py
│   └── data/20260604_154130_348_solver_throughput_bar/best_solver_summary.csv
│       data/ue_density_reference/ue_count_best_solver_summary.csv
├── fig03_tts/
│   ├── plot_fig3_tts_table_bar.py
│   └── data/20260604_160259_375_solver_tts_rbk_600/tts_rbk_summary.csv
├── fig04_random_ue_tts/
│   ├── plot_fig4_ieee_normalized_tts.py
│   └── data/20260605_100244_666_ap30_ue_seed_tts_throughput/ap30_ue_seed_tts_summary.csv
│       data/20260605_100244_666_ap30_ue_seed_tts_throughput/experiment_config.yaml
├── fig05_area_size/
│   ├── plot_fig5_ieee_normalized_max_dist.py
│   └── data/20260605_171908_796_max_dist_solver_throughput/max_dist_best_solver_summary.csv
│       data/20260605_171908_796_max_dist_solver_throughput/max_dist_best_solver_summary_plotted.csv
└── result_picture/                      output folder; created at run time
```

## Figure ↔ script ↔ data

| Manuscript figure | Script | Input data |
|---|---|---|
| Fig. 1 — dense WLAN topology (20 APs, 200 UEs, 150 m × 150 m) | `fig01_topology_model/fig1_wlan_topology_plot.py` | none. Positions are generated procedurally from the seeds in `configs/default_config.yaml` (`ap_seed = 16`, `ue_seed = 17`, 10 UEs per AP, 150 m box, 70 m carrier-sense range) together with `core/environment.py` |
| Fig. 2 — normalized throughput vs. number of APs | `fig02_ap_throughput/plot_fig2_ieee_line.py` | `best_solver_summary.csv` (5 AP counts × 7 solvers) and `ue_count_best_solver_summary.csv` (see *Data provenance*) |
| Fig. 3 — time-to-solution vs. number of APs | `fig03_tts/plot_fig3_tts_table_bar.py` | `tts_rbk_summary.csv` (5 AP counts × 6 solvers) |
| Fig. 4 — TTS vs. normalized throughput over 20 random UE topologies | `fig04_random_ue_tts/plot_fig4_ieee_normalized_tts.py` | `ap30_ue_seed_tts_summary.csv` (20 UE seeds × 6 solvers) and `experiment_config.yaml` |
| Fig. 5 — normalized throughput vs. AP count and deployment area | `fig05_area_size/plot_fig5_ieee_normalized_max_dist.py` | `max_dist_best_solver_summary_plotted.csv` (4 area sizes × 5 AP counts × 7 solvers) |

## How to run

```bash
python fig01_topology_model/fig1_wlan_topology_plot.py
python fig02_ap_throughput/plot_fig2_ieee_line.py
python fig03_tts/plot_fig3_tts_table_bar.py
python fig04_random_ue_tts/plot_fig4_ieee_normalized_tts.py
python fig05_area_size/plot_fig5_ieee_normalized_max_dist.py
```

No command-line arguments are needed. Each run creates a timestamped folder
`result_picture/<YYYYmmdd_HHMMSS_mmm>_<figure>/`, so repeated runs never overwrite each other;
Figs. 2–5 additionally accept `--output-dir <path>`.

Files written per figure:

| Figure | Output files |
|---|---|
| Fig. 1 | `wlan_topology_ap20_apseed16_ueseed17_ieee.{png,pdf}` — the script renders only the 20-AP panel used in the manuscript |
| Fig. 2 | `fig2_ieee_ap_throughput_line.{png,pdf,eps}` and the intermediate table `fig2_ap30_replaced_with_fig5_ue10_summary.csv` |
| Fig. 3 | `fig3_tts_table_bar.{png,pdf,eps}` |
| Fig. 4 | `fig4_ieee_normalized_throughput_tts.{png,pdf,eps}` |
| Fig. 5 | `fig5_ieee_normalized_max_dist.{png,pdf,eps}` and the four single panels `fig5_1..4.{png,pdf,eps}` |

All five commands were executed on the data shipped here: each exits with status 0, and the PNG it
produces matches the submitted figure — Figs. 1–4 byte-identical, Fig. 5 pixel-identical.

## Environment

```
Python      3.10.20
matplotlib  3.10.8
numpy       2.2.5
pandas      2.3.3
PyYAML      6.0.3
```

`requirements.txt` pins the same packages. The scripts request the **Arial** font and fall back to
DejaVu Sans if it is unavailable (only text rasterization is affected, not the data). With matplotlib
3.10.0 the same figures are produced to within 0.3 % of pixels (text and line anti-aliasing only).

## Data provenance

* **Fig. 2.** `best_solver_summary.csv` contains the normalized throughput per AP count and solver. The 
script stops with an error if that file is
  missing; the substituted table is written out as `fig2_ap30_replaced_with_fig5_ue10_summary.csv`.
* **Fig. 3.** The success criterion is the throughput ratio with respect to the best CIM sample of the
  same scenario (`R_alg / R_bk > 0.95`), as recorded in the CSV column `rbk_definition`. Solvers with
  `p_success = 0` (SA in every scenario) have `tts_ms = inf` and are omitted from the figure.
* **Fig. 4.** The y-axis is normalized at plot time as `best throughput / theoretical maximum`, where the
  theoretical maximum is recomputed from the environment parameters stored in `experiment_config.yaml`;
  that file is therefore a required input alongside the summary CSV.
* **Fig. 5.** `max_dist_best_solver_summary.csv` is the raw measurement of the area-size sweep.
  `max_dist_best_solver_summary_plotted.csv` is the table that is actually plotted.

## Note on the scripts

The plotting logic, styles and numerical computations are those used for the manuscript. The output is written to
`result_picture/<timestamp>_<figure>/` instead of into the data folder. The `data/` folders are
read-only inputs.
