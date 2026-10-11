using LinearAlgebra
using SparseArrays
using Arpack
using ForwardDiff

# Same-module includes (no module barriers). Three pieces: operators, cutoff, krylov.
# Include only one miisoN.jl per Julia module — each defines its own R_C / R_D / theta_wedge.
include(joinpath(@__DIR__, "operators.jl"))
include(joinpath(@__DIR__, "cutoff.jl"))
include(joinpath(@__DIR__, "krylov.jl"))
