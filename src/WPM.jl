using Sobol
using Roots

using Random
using Distributions
using SparseArrays
using LinearAlgebra
using NFFT
# using LoopVectorization # I did ``export JULIA_NUM_THREADS=auto`` in the terminal (not Julia REPL)
# using InteractiveUtils # Import to use ``@code_warntype`` 



"""
Describe meta particules, represented by a Dirac distribution in (``x``, ``v``), with a weight ``β``
"""
struct Particles{T}
    x::Array{T,2}    # list of the positions
    v::Array{T,2}    # list of the velocities
    β::Vector{T}      # list of the weights of the particules
    cplxβ::Vector{Complex{T}}      # list of the weights of the particules
    nbpart::Int       # number of particules
    dim::Int
    type

    function Particles{T}(x, v, β) where {T<:Real}
        new(x, v, β, convert.(Complex{T}, β), size(x)[2], size(x)[1], T)
    end
end

"""
    Defines Runge-Kutta-Nystrom time integrator via its Butcher tableau,
    and holds some pre-allocated arrays used for the time integration only.
"""
struct symplectic_rkn4{T<:Real}
    a::Array{T,2}
    b̄::Vector{T}
    c::Vector{T}
    b::Vector{T}
    dt::T
    fg::Array{Array{T,2},1}
    G::Array{T,2}
    nb_steps::Int

    function symplectic_rkn4{T}(X, dt) where {T<:Real}
        # a, b̄, c, b correspond to the Butcher tableau of Runge-Kutta-Nystrom 3steps order4.
        a = [0.0 0.0 0.0
            (2-√3)/12 0.0 0.0
            0.0 √(3)/6 0.0]
        b̄ = [(5 - 3 * √3) / 24, (3 + √3) / 12, (1 + √3) / 24]
        c = [(3 + √3) / 6, (3 - √3) / 6, (3 + √3) / 6]
        b = [(3 - 2 * √3) / 12, 1 / 2, (3 + 2 * √3) / 12]

        stages = 3

        new(a .* dt^2, b̄ .* dt^2, c .* dt, b .* dt, dt,
            [zeros(size(X)) for _ = 1:stages],  # fg
            similar(X), # G
            stages # number of steps
        )
    end
end


struct rkn4{T<:Real}
    a::Array{T,2}
    b̄::Vector{T}
    c::Vector{T}
    b::Vector{T}
    dt::T
    fg::Array{Array{T,2},1}
    G::Array{T,2}
    nb_steps::Int

    function rkn4{T}(X, dt) where {T<:Real}
        # a, b̄, c, b correspond to the Butcher tableau of Runge-Kutta-Nystrom 3steps order4.
        a = [0.0 0.0 0.0
            1/8 0.0 0.0
            0.0 1/2 0.0]
        b̄ = [1 / 6, 1 / 3, 0]
        c = [0, 1 / 2, 1]
        b = [1 / 6, 4 / 6, 1 / 6]

        stages = 3

        new(a .* dt^2, b̄ .* dt^2, c .* dt, b .* dt, dt,
            [zeros(size(X)) for _ = 1:stages],  # fg
            similar(X), # G
            stages # number of steps
        )
    end
end


struct rkn5{T<:Real}
    a::Array{T,2}
    b̄::Vector{T}
    c::Vector{T}
    b::Vector{T}
    dt::T
    fg::Array{Array{T,2},1}
    G::Array{T,2}
    nb_steps::Int

    function rkn5{T}(X, dt) where {T<:Real}
        # a, b̄, c, b correspond to the Butcher tableau of Runge-Kutta-Nystrom 3steps order4.
        a = [0.0 0.0 0.0 0.0
            1/50 0.0 0.0 0.0
            -1/27 7/27 0.0 0.0
            3/10 -2/35 9/35 0.0]
        b̄ = [14 / 336, 100 / 336, 54 / 336, 0.0]
        c = [0, 1 / 5, 2 / 3, 1.0]
        b = [14 / 336, 125 / 336, 162 / 336, 35 / 336]

        stages = 4
        new(a .* dt^2, b̄ .* dt^2, c .* dt, b .* dt, dt,
            [zeros(size(X)) for _ = 1:stages],  # fg
            similar(X), # G
            stages # number of steps
        )
    end
end


"""
Holds pre-allocated arrays
"""
struct ParticleMover{T<:Real,DIM}
    ∂Φ::Array{T,2}
    torus::Vector{T}
    torus_size::T
    kxs::Vector{T}
    K::Int
    C::Array{T,DIM}
    S::Array{T,DIM}
    computedyet::BitArray{DIM}
    tmpsinkcosk::Array{T,2}
    # Vérifier les dimensions :
    nbfouriermodes::Int64
    fourier_coeffs_∂rho::Array{ComplexF64,1}
    approx_rho::Array{ComplexF64,1}
    invks::Array{ComplexF64,1}
    plan_invfft::NFFTPlan{Float64,1,1}
    ###########################
    Eelec²tot²::Vector{T} # Holds Eelec² and Etot²
    momentum::Array{T,1}
    rkn::symplectic_rkn4{T}
    dt::T
    type
    dim::Int64

    function ParticleMover{T,DIM}(particles::Particles, torus, K, dt) where {T<:Real,DIM}
        ∂Φ = similar(particles.x)
        tmpsinkcosk = Array{T,2}(undef, 2, particles.nbpart)

        # NFFT: Only works in 1D for now
        nbfouriermodes = 2K
        fourier_coeffs_∂rho = Array{ComplexF64,1}(undef, nbfouriermodes)
        approx_rho = Array{ComplexF64,1}(undef, particles.nbpart)

        invks = Vector{ComplexF64}(undef, nbfouriermodes)
        invks .= collect(-nbfouriermodes/2:(nbfouriermodes-1)/2)
        invks[Int64(nbfouriermodes / 2 + 1)] = 1
        invks .*= 2π / torus[1]
        invks = invks ./ (invks .^ 2)
        invks[Int64(nbfouriermodes / 2 + 1)] = 0

        ε = 1e-12
        invfft = plan_nfft(particles.x ./ torus[1] .- 0.5, (nbfouriermodes,); reltol=ε)
        ##########################

        new(∂Φ,
            convert.(T, torus), convert(T, *(torus...)), convert.(T, 2π ./ torus), K,
            Array{T,DIM}(undef, fill(2K + 1, DIM)...), #C
            Array{T,DIM}(undef, fill(2K + 1, DIM)...), #S
            BitArray{DIM}(undef, fill(2K + 1, DIM)...), # computedyet
            tmpsinkcosk,
            nbfouriermodes, fourier_coeffs_∂rho, approx_rho, invks,
            invfft,
            zeros(T, 2), # Eelec²tot² (tuple)
            similar(particles.v[:, 1]), # momentum
            symplectic_rkn4{T}(particles.x, dt), # rkn
            # rkn4{T}(particles.x, dt), # rk4
            # rkn5{T}(particles.x, dt),
            dt, T, DIM)
    end
end



#==== Time steppers ====#
"""RKN_timestepper!(p, pmover, kernel)
"""
function RKN_timestepper!(p, pmover; kernel=kernel_poisson!)
    @views begin
        for s = 1:pmover.rkn.nb_steps
            @. pmover.rkn.G = p.x + p.v * pmover.rkn.c[s]

            for ss = 1:pmover.rkn.nb_steps
                @. pmover.rkn.G += pmover.rkn.a[s, ss] * pmover.rkn.fg[ss]
            end
            kernel(pmover.rkn.fg[s], pmover.rkn.G, p, pmover)
        end

        @. p.x += pmover.dt * p.v
        for s = eachindex(pmover.rkn.b̄)
            @. p.x += pmover.rkn.b̄[s] * pmover.rkn.fg[s]
            @. p.v += pmover.rkn.b[s] * pmover.rkn.fg[s]
        end

        kernel(pmover.∂Φ, p.x, p, pmover) # only useful to update C, S with new positions
    end
end


"""strang_splitting!(particles, pmover, kernel)  
    
    Other method for advecting (X, V) on a time step. 

    Uses Verlet scheme (of order 2).

    Args:
    - particles: Particle struct
    - pmover: ParticleMover struct
    - kernel: how to compute the flow field.

    Updates X, V in place, and returns coefficients C, S at current time. 
"""
function splitting!(particles, pmover, kernel; order=2)
    if order == 2
        @. particles.v += pmover.∂Φ * pmover.dt / 2
        @. particles.x += particles.v * pmover.dt
        kernel(pmover.∂Φ, particles.x, particles, pmover)
        @. particles.v += pmover.∂Φ * pmover.dt / 2
    elseif order == 6
        w1 = -0.117767998417887E1
        w2 = 0.235573213359537E0
        w3 = 0.784513610477560E0
        w0 = 1 - 2(w1 + w2 + w3)
        # A step of second order splitting
        @. particles.v += pmover.∂Φ * w3 * pmover.dt / 2
        @. particles.x += particles.v * w3 * pmover.dt
        kernel(pmover.∂Φ, particles.x, particles, pmover)
        @. particles.v += pmover.∂Φ * (w3 + w2) * pmover.dt / 2
        @. particles.x += particles.v * w2 * pmover.dt
        kernel(pmover.∂Φ, particles.x, particles, pmover)
        @. particles.v += pmover.∂Φ * (w2 + w1) * pmover.dt / 2
        @. particles.x += particles.v * w1 * pmover.dt
        kernel(pmover.∂Φ, particles.x, particles, pmover)
        @. particles.v += pmover.∂Φ * (w0 + w1) * pmover.dt / 2
        @. particles.x += particles.v * w0 * pmover.dt
        kernel(pmover.∂Φ, particles.x, particles, pmover)
        @. particles.v += pmover.∂Φ * (w0 + w1) * pmover.dt / 2
        @. particles.x += particles.v * w1 * pmover.dt
        kernel(pmover.∂Φ, particles.x, particles, pmover)
        @. particles.v += pmover.∂Φ * (w2 + w1) * pmover.dt / 2
        @. particles.x += particles.v * w2 * pmover.dt
        kernel(pmover.∂Φ, particles.x, particles, pmover)
        @. particles.v += pmover.∂Φ * (w3 + w2) * pmover.dt / 2
        @. particles.x += particles.v * w3 * pmover.dt
        kernel(pmover.∂Φ, particles.x, particles, pmover)
        @. particles.v += pmover.∂Φ * w3 * pmover.dt / 2
    else
        throw(ArgumentError("The argument ``order'' can be either 2 or 6."))
    end
end



# ===== Kernel computations ==== #
"""
Compute -∂_x Φ[f](`x`) and stores it in `dst`.
"""
function kernel_poisson!(dst, x, p, pmover)
    dst .= zero(pmover.type)

    pmover.S .= zero(p.type)
    pmover.C .= zero(p.type)
    pmover.computedyet .= false

    @views for idxk = CartesianIndices(pmover.C)
        k = idxk.I .- (pmover.K + 1)
        ξk = k .* pmover.kxs
        normξk² = sum(ξk .^ 2)
        if (normξk² == 0) || (sum(abs.(k)) > pmover.K)
            pmover.computedyet[idxk] = true
        end
        pmover.computedyet[idxk] && continue

        idxminusk = CartesianIndex(.-k .+ (pmover.K + 1))

        skck = compute_Sₖ_Cₖ!(pmover.tmpsinkcosk, x, ξk, p.β)
        pmover.computedyet[idxk] = true
        pmover.computedyet[idxminusk] = true
        pmover.S[idxk] = skck[1]
        pmover.S[idxminusk] = -skck[1]
        pmover.C[idxk] = skck[2]
        pmover.C[idxminusk] = skck[2]

        # The line below computes -∂Φ[f](`x`) and stores it to `dst`. 
        # Changing dynamics : 
        #   "+=": repulsive potential (plasmas dynamics)
        #   "-=": attractive potential (galaxies dynamics)
        for pcol = 1:size(dst)[2]
            for prow = 1:size(dst)[1]
                @inbounds dst[prow, pcol] += 2 * (pmover.C[idxk] * pmover.tmpsinkcosk[1, pcol] -
                                                  pmover.S[idxk] * pmover.tmpsinkcosk[2, pcol]) * ξk[prow] / normξk²
            end
        end
    end

    dst ./= pmover.torus_size
end
#
#
function compute_Sₖ_Cₖ!(tmpsinkcosk, x, ξ, β)
    S = 0.0
    C = 0.0
    for (idx, xcol) = enumerate(eachcol(x))
        skck = sincos(dot(xcol, ξ))
        @inbounds tmpsinkcosk[:, idx] .= skck
        @inbounds S += skck[1] * β[idx]
        @inbounds C += skck[2] * β[idx]
    end
    return S, C
end


function kernel_gyrokinetic!(dst, x, p, pmover)
    @. dst = -x
end


function kernel_freestreaming!(dst, x, p, pmover)
    dst .= zero(pmover.type)

    pmover.S .= zero(pmover.type)
    pmover.C .= zero(pmover.type)

    @views for idxk = CartesianIndices(pmover.C)
        k = idxk.I .- (pmover.K + 1)
        ξk = k .* pmover.kxs
        normξk² = sum(ξk .^ 2)
        (normξk² == 0 || sum(abs.(k)) > pmover.K) && continue

        for (idx, xcol) = enumerate(eachcol(x))
            skck = sincos(dot(xcol, ξk))
            @inbounds pmover.tmpsinkcosk[:, idx] .= skck
            @inbounds pmover.S[idxk] += skck[1] * p.β[idx]
            @inbounds pmover.C[idxk] += skck[2] * p.β[idx]
        end
    end
end


function kernel_poisson_nfft!(dst, x, p, pmover)
    ###
    # Works only in 1D for now
    ###

    for idp = 1:p.nbpart
        for d = 1:p.dim
            x[d, idp] >= pmover.torus[d] ? while (x[d, idp] >= pmover.torus[d])
                x[d, idp] -= pmover.torus[d]
            end : nothing
            x[d, idp] < 0 ? while (x[d, idp] < 0)
                x[d, idp] += pmover.torus[d]
            end : nothing
        end
    end

    # Forward NFFT:
    nodes!(pmover.plan_invfft, reshape(x[1, :] ./ pmover.torus[1] .- 0.5, 1, :))
    pmover.fourier_coeffs_∂rho .= adjoint(pmover.plan_invfft) * p.cplxβ

    # Inverse NFFT:
    pmover.fourier_coeffs_∂rho .*= pmover.invks * -1im
    pmover.approx_rho .= pmover.plan_invfft * pmover.fourier_coeffs_∂rho

    index_freq_zero = convert(Int64, pmover.nbfouriermodes / 2 + 1)
    for k = 0:(index_freq_zero-1)
        idxk = k + index_freq_zero
        idxminusk = -k + index_freq_zero
        # sk, ck are inverted because of the multiplication by -1im:
        sminusk, cminusk = reim(pmover.fourier_coeffs_∂rho[idxminusk]) .* (2π / pmover.torus[1])
        cminusk = -cminusk
        pmover.S[idxk] = -sminusk
        pmover.S[idxminusk] = sminusk
        pmover.C[idxk] = cminusk
        pmover.C[idxminusk] = cminusk
    end

    dst .= -real.(pmover.approx_rho') ./ pmover.torus_size
end




# ===== Some quantities we can compute at each step ==== #
"""compute_electricalenergy²(p, pmover)

    Returns the square of the electrical energy
"""
function compute_electricalenergy²!(p, pmover)
    pmover.Eelec²tot²[1] = 0.0
    @views for idxk = CartesianIndices(pmover.C)
        k = idxk.I .- (pmover.K + 1)
        ξk = k .* pmover.kxs
        normξk² = sum(ξk .^ 2)
        (normξk² == 0 || sum(abs.(k)) > pmover.K) && continue
        @inbounds pmover.Eelec²tot²[1] += (pmover.C[idxk]^2 + pmover.S[idxk]^2) / normξk²
    end
    pmover.Eelec²tot²[1] /= pmover.torus_size
end

function compute_momentum!(particles, pmover)
    pmover.momentum .= zero(pmover.type)
    for (idv, vv) = enumerate(eachcol(particles.v))
        @inbounds pmover.momentum .+= vv .* particles.β[idv]
    end
end

function compute_totalenergy²!(particles, pmover)
    pmover.Eelec²tot²[2] = zero(pmover.type)
    @views for (idv, vv) = enumerate(eachcol(particles.v))
        @inbounds pmover.Eelec²tot²[2] += sum(abs2, vv) * particles.β[idv]
    end
    pmover.Eelec²tot²[2] += pmover.Eelec²tot²[1]
    pmover.Eelec²tot²[2] /= 2
end



""" periodic_boundary_conditions!(p, pmover)

    Impose periodic boundary conditions in space.
"""
function periodic_boundary_conditions!(p, pmover)
    @inbounds begin
        @views for idp = 1:p.nbpart
            for d = 1:p.dim
                if p.x[d, idp] >= pmover.torus[d]
                    while p.x[d, idp] >= pmover.torus[d]
                        p.x[d, idp] -= pmover.torus[d]
                    end
                elseif p.x[d, idp] < 0
                    while p.x[d, idp] < 0
                        p.x[d, idp] += pmover.torus[d]
                    end
                end
            end
        end
    end
end



function WPM_step!(p, pmover; kernel=kernel_poisson!)
    RKN_timestepper!(p, pmover; kernel)
    # splitting!(p, pmover, kernel, order=2) # order=2 or 6

    periodic_boundary_conditions!(p, pmover)

    compute_momentum!(p, pmover)
    compute_electricalenergy²!(p, pmover)
    compute_totalenergy²!(p, pmover)

    return nothing
end