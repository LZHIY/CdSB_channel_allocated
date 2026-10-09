# Experiment code — data generation for all figures and tables of the manuscript

This folder contains the simulation code that produces **every measurement reported in the manuscript**
(five figures and five tables). The plotting scripts and the plotted data are provided separately in the
sibling folder `../fig_plot_and_data/`.

The scripts sit in the root of this folder on purpose: each of them resolves `configs/`, `core/` and
`solvers/` relative to its own location, so every script runs with no arguments and no path setup. The 20
code files are byte-identical to the code used to produce the results.

## Contents

```
experiment_code/
├── configs/default_config.yaml                  simulation defaults (read by the Fig. 1 script)
├── core/environment.py                          WLAN cross-layer model (PHY / MAC / CSMA-CA, Ising & QUBO construction)
├── core/base_solver.py                          solver base class
├── solvers/sa_solver.py                         simulated annealing
├── solvers/greedy_solver.py                     greedy channel assignment
├── solvers/cim_solver.py                        coherent Ising machine (parallel)
├── solvers/dsb_solver.py                        discrete simulated bifurcation (dSB)
├── solvers/bsb_solver.py                        ballistic simulated bifurcation (bSB)
├── solvers/gbsb_solver.py                       generalized bSB (GbSB)
├── solvers/gdsb_solver.py                       CdSB, the proposed method (stored as GDSB)
├── solvers/random_solver.py                     random baseline (not used in the manuscript)
├── fig1_wlan_topology_plot.py                   Fig. 1 — topology illustration
├── fig2_solver_throughput_bar_experiment.py     Fig. 2 — normalized throughput vs. number of APs
├── fig3_solver_tts_rbk_experiment.py            Fig. 3 and Table III — TTS, success rate, single-run time
├── fig4_ap30_ue_seed_tts_throughput.py          Fig. 4 — robustness over 20 random UE topologies
├── fig5_ue_count_solver_throughput_experiment.py  Table IV — UE/AP ratio sweep
├── max_dist_solver_throughput_experiment.py     Fig. 5 — AP count × deployment area size
├── recompute_metrics_sumS.py                    Table II (ablation), Table 2 (classical baselines), surrogate checks of Sec. II.D
└── sensitivity_sweep.py                         Table V — parameter sensitivity
```

## Paper item ↔ script ↔ output ↔ runtime

| Paper item | Script | Output (new timestamped folder per run) | Runtime* |
|---|---|---|---|
| Fig. 1 — topology | `fig1_wlan_topology_plot.py` | `result_picturre/fig1_wlan_topology_plot/<ts>_wlan_topology/wlan_topology_ap{10,20,30,40,50}_apseed16_ueseed17_ieee.{png,pdf}` | seconds |
| Fig. 2 — throughput vs. AP count | `fig2_solver_throughput_bar_experiment.py` | `resultsTHROUGHT/<ts>_solver_throughput_bar/`: `raw_solver_runs.csv`, `best_solver_summary.csv`, `experiment_config.yaml`, diagnostic PNG/PDF | ≈ 10 min |
| Fig. 3 + Table III — TTS, success rate, single-run time | `fig3_solver_tts_rbk_experiment.py` | `resultsTTS/<ts>_solver_tts_rbk_600/`: `tts_rbk_summary.csv`, `tts_rbk_raw_600_results.csv`, `tts_rbk_by_ap.csv`, `experiment_config.yaml`, diagnostic PNG/PDF | ≈ 5 min |
| Fig. 4 — random-UE robustness | `fig4_ap30_ue_seed_tts_throughput.py` | `resultsUESeedTTS/<ts>_ap30_ue_seed_tts_throughput/`: `ap30_ue_seed_tts_summary.csv`, `ap30_ue_seed_raw_solver_samples.csv`, `ap30_ue_seed_rbk.csv`, `experiment_config.yaml`, diagnostic PNG/PDF | ≈ 15 min |
| Table IV — UE/AP ratio | `fig5_ue_count_solver_throughput_experiment.py` | `resultsUECountThroughput/<ts>_ue_count_solver_throughput/`: `ue_count_best_solver_summary.csv`, `ue_count_raw_solver_runs.csv`, `experiment_config.yaml`, diagnostic PNG/PDF | ≈ 32 min |
| Fig. 5 — AP count × deployment area | `max_dist_solver_throughput_experiment.py` | `resultsMaxDistThroughput/<ts>_max_dist_solver_throughput/`: `max_dist_best_solver_summary.csv`, `max_dist_raw_solver_runs.csv`, `experiment_config.yaml`, diagnostic PNG/PDF | ≈ 13 min |
| Table II — ablation; Table 2 — classical baselines; surrogate checks of Sec. II.D | `recompute_metrics_sumS.py` | `E1_correlation_sumS.csv`, `E3_ablation_sumS.csv`, `E4_baselines_sumS.csv` written to the working directory (E2 prints to stdout) | E4 ≈ 15 min (dominated by the tabu-search baseline, 840 s); E1–E3 depend on `--samples` and `--trials` |
| Table V — parameter sensitivity | `sensitivity_sweep.py` | `sensitivity_results.csv` in the working directory | ≈ 27 min (21 parameter values × 3 repetitions) |

\* Solver wall-clock totals measured on the machine used for the paper (Python, Intel Core Ultra X7358H
@ 1.90 GHz, 32 GB RAM).

## How to run

```bash
# paper configuration (the defaults of each script are the settings used for the reported results)
python fig1_wlan_topology_plot.py
python fig2_solver_throughput_bar_experiment.py
python fig3_solver_tts_rbk_experiment.py
python fig4_ap30_ue_seed_tts_throughput.py
python fig5_ue_count_solver_throughput_experiment.py
python max_dist_solver_throughput_experiment.py
python recompute_metrics_sumS.py            # all checks E1–E4
python recompute_metrics_sumS.py --only E1  # correlation check only (fastest)
python sensitivity_sweep.py

# smoke test: small scale, verifies the pipeline, not the paper configuration
python fig2_solver_throughput_bar_experiment.py --quick
python fig3_solver_tts_rbk_experiment.py --quick
python fig4_ap30_ue_seed_tts_throughput.py --quick
python fig5_ue_count_solver_throughput_experiment.py --quick
python max_dist_solver_throughput_experiment.py --quick
python sensitivity_sweep.py --quick
```

Every script supports `--help`. Common options include `--ap-list`, `--solvers`, `--samples`, `--runs`,
`--dt`, `--steps` and `--num-agents`. The six scripts that write run folders create a new timestamped
folder on each run, containing the per-run records (`*_raw_solver_runs.csv`, and per-sample records where
applicable), the table used for plotting (`*_best_solver_summary.csv`) and a snapshot of the parameters
(`experiment_config.yaml`). `recompute_metrics_sumS.py` and `sensitivity_sweep.py` write their CSV files to
the current working directory, so they are best run from this folder.

## Configuration used for the reported results

| Script | Key parameters (script defaults) |
|---|---|
| `fig1_wlan_topology_plot.py` | from `configs/default_config.yaml`: `ap_list = [10,20,30,40,50]`, `num_ch = 3`, `num_ue_per_ap = 10`, `ap_seed = 16`, `ue_seed = 17`, `max_dist = 150`, `carrier_sense_range_m = 70` |
| `fig2_solver_throughput_bar_experiment.py` | AP = 10/20/30/40/50, `num_ch = 3`, 10 UEs per AP, `ap_seed = 15`, `ue_seed = 16`, `dt = 0.15`, `steps = 900`, `num_agents = 300`, 1 run per solver; CdSB/GbSB strength `A = {10: 0.30, 20: 0.48, 30: 0.39, 40: 0.56, 50: 0.63}` |
| `fig3_solver_tts_rbk_experiment.py` | same AP list, topology seeds and `num_ch`; `samples = 100`, `dt = 0.15`, `steps = 600`, `num_agents = 100`, `p_target = 0.99`, success criterion `R_alg / R_bk > 0.95`; `A = {10: 0.30, 20: 0.49, 30: 0.39, 40: 0.59, 50: 0.63}` |
| `fig4_ap30_ue_seed_tts_throughput.py` | `num_ap = 30` (300 UEs), `ap_seed = 15`, 20 UE seeds (`16, 19, 20, 34, 234, 432, 1, 213, 343, 453, 545, 654, 73, 85, 933, 1234, 2345, 3545, 6765, 6785`), `samples = 100`, `dt = 0.15`, `steps = 600`, `num_agents = 100`, `A = 0.39` |
| `fig5_ue_count_solver_throughput_experiment.py` | `num_ap = 30`, UE/AP = 10/15/20/25/30, `ap_seed = 15`, `ue_seed = 16`, `max_dist = 150`, `dt = 0.15`, `steps = 900`, `num_agents = 100`, 1 run per solver |
| `max_dist_solver_throughput_experiment.py` | area size L = 150/200/250/300 m, AP = 10…50, `ap_seed = 15`, `ue_seed = 16`, `dt = 0.15`, `steps = 900`, `num_agents = 100`, 1 run per solver; `A = {10: 0.30, 20: 0.49, 30: 0.39, 40: 0.59, 50: 0.63}` |
| `recompute_metrics_sumS.py` | E1/E2 (`M = 20`, UE/AP = 5, `samples = 3000`); E3 ablation (UE/AP = 10); E4 classical baselines (UE/AP = 5); common: `agents = 100`, `steps = 2000`, `dt = 0.05`, `A = 0.32`, `zeta = 0.6`, `gamma = 0.1`, `V1 = 1.0`, `trials = 10`, `seed = 0`, `ls_iters = 300`, `ts_iters = 150`, `ts_tabu = 7` |
| `sensitivity_sweep.py` | base configuration `M = 30`, `K = 3`, UE/AP = 10, `ap_seed = 15`, `ue_seed = 16`, `max_dist = 150`, `A = 0.39`, `dt = 0.15`, `steps = 900`, `agents = 300`, `zeta = 0.6`, `gamma = 0.1`, `V1 = 1.0`, `V2 = 100`, `d_cs = 70`; sweeps `A = 0.29–0.49`, `dt = 0.05–0.20`, `steps = 600/900/1200`, `V2 = 50/100/200/500`, `d_cs = 50–90 m`, 3 repetitions per point |

`configs/default_config.yaml` only supplies the defaults needed by the Fig. 1 script; for all other
scripts the key settings are the constants listed above.

## Environment

Python 3.10 or newer with `numpy`, `pandas`, `PyYAML` and `matplotlib` (see `requirements.txt`).
The environment used for the reported results was Python 3.10.20, numpy 2.2.5, pandas 2.3.3,
PyYAML 6.0.3, matplotlib 3.10.8. Messages printed by some scripts are in Chinese; on a console that is
not UTF-8 they may appear garbled (set `PYTHONIOENCODING=utf-8`), which does not affect the computation.

## Data reuse across experiments

The `M = 30`, `UE/AP = 10` measurement of the UE-density experiment (`fig5_ue_count_solver_throughput_experiment.py`)
is also used when plotting Fig. 2 (its AP = 30 rows) and in the `L = 150 m` panel of Fig. 5, so that the
same physical scenario is reported consistently across the manuscript. The reuse is documented in the
sibling folder `../fig_plot_and_data/`.
