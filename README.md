# MIISO 

Companion code for the MIISO paper.

## Layout

```
src/        MIISO2–6 and the generalized-alpha code
test/       tests for the family
scripts/    paper figures
```

## `src/`

Include one of `miiso2.jl` … `miiso6.jl`. That pulls in the shared pieces (`operators`, `cutoff`, `krylov`).

```julia
include("src/miiso4.jl")
yT = miiso4_solve(f, y0, T, h; setup=setup, J=J, ft=ft)
```

Only include one `miisoN.jl` per module.

## `test/`

| Path | Writes |
|------|--------|
| `membrance/convergence.jl` | convergence errors |
| `membrance/cost_to_error.jl` | cost vs error |
| `membrance/scaling.jl` | mesh scaling |
| `membrance/outlier_visual/run.jl` | MIISO4 vs exprb fields |
| `energies/run.jl` | band energies |
| `cutoff_tracking/run.jl` | gap / leakage |
| `bridge/run.jl` | arch crown history |

```bash
julia --project=. --startup-file=no test/membrance/convergence.jl
```

## `scripts/`

| Script | Figures |
|--------|---------|
| `fig_pade_repair.py` | 1–4 |
| `fig_taper.py` | taper |
| `fig_miiso_surfaces.py` | 6 |
| `fig_modal_gallery.py` | 7–12 |
| `fig_bridge_snapshots.py` | bridge |
| `fig_corner_blowup.py` | 16–17 |

`scripts/codes/` holds the spectral maps those plots need.

### Disclaimer

The figure scripts are mostly AI assisted. They are mechanical plotting. Whatever could be messed up was verified. The method itself is in `src/`. The tests and source are manually written, but with AI for assistance and prototyping, polishing and refinement. 
