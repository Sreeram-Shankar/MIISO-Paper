using LinearAlgebra
using SparseArrays
using Arpack
using ForwardDiff

#these three files are shared by every miiso method.
#operators holds JVP and workspaces, cutoff holds the outlier
#eigenpair cache, krylov holds Arnoldi and the phi actions.
#only one miiso(N).jl should be included per Julia module, because
#each one defines its own custom amplification maps.
include(joinpath(@__DIR__, "operators.jl"))
include(joinpath(@__DIR__, "cutoff.jl"))
include(joinpath(@__DIR__, "krylov.jl"))
