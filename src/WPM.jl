using Sobol
using Roots

using Random
using Distributions
using SparseArrays
using LinearAlgebra
using InteractiveUtils # Import to use ``@code_warntype`` 



"""
Describe meta particules, represented by a Dirac distribution in (``x``, ``v``), with a weight ``β``
"""
struct Particles{T}
    x :: Array{T, 2}    # list of the positions
    v :: Array{T, 2}    # list of the velocities
    β :: Vector{T}      # list of the weights of the particules
    nbpart :: Int       # nmber of particules
end

"""
    Defines Runge-Kutta-Nystrom time integrator via its Butcher tableau,
    and holds some pre-allocated arrays used for the time integration only.
"""
struct symplectic_rkn4{T<:Real}
    a ::    Array{T, 2}
    b̄ ::    Vector{T}
    c ::    Vector{T}
    b ::    Vector{T}
    dt ::   T
    fg ::   Array{Array{T, 2}, 1}
    G ::    Array{T, 2}
    nb_steps :: Int

    function symplectic_rkn4{T}(X, dt) where T<:Real
        # a, b̄, c, b correspond to the Butcher tableau of Runge-Kutta-Nystrom 3steps order4.
        a = [0.0        0.0         0.0; 
            (2-√3)/12   0.0         0.0; 
            0.0         √(3)/6      0.0]
        b̄ = [(5 - 3*√3)/24,     (3+√3)/12,  (1+√3)/24]
        c = [(3+√3)/6,          (3-√3)/6,   (3+√3)/6]
        b = [(3-2*√3)/12,       1/2,        (3+2*√3)/12]

        stages = 3

        new(a .* dt^2, b̄ .* dt^2, c .* dt, b .* dt, dt, 
            [similar(X) for _=1:stages],  # fg
            similar(X), # G
            stages # number of steps
        )
    end
end


struct rkn5{T<:Real}
    a :: Array{T, 2}
    b̄ :: Vector{T}
    c :: Vector{T}
    b :: Vector{T}
    dt :: T
    fg :: Array{T, 2}
    G ::  Array{T, 1}
    nb_steps :: Int

    function rkn5{T}(X, dt) where T<:Real
        # a, b̄, c, b correspond to the Butcher tableau of Runge-Kutta-Nystrom 3steps order4.
        a = [0.0        0.0         0.0     0.0; 
            1/50        0.0         0.0     0.0; 
            -1/27       7/27        0.0     0.0;
            3/10        -2/35       9/35    0.0]
        b̄ = [14/336,    100/336,    54/336, 0.0]
        c = [0,         1/5,        2/3,    1.0]
        b = [14/336,    125/336,    162/336, 35/336]

        stages = 4
        new(a .* dt^2, b̄ .* dt^2, c .* dt, b .* dt, dt, 
            zeros(T, length(X), stages),  # fg
            similar(X, T), # G
            stages
        )
    end
end


"""
Holds pre-allocated arrays
"""
struct ParticleMover{T<:Real}
    Φ :: Array{T, 2}
    ∂Φ :: Array{T, 2}
    meshx :: OneDGrid{T}
    torus_size :: T
    kx :: T
    K :: Int
    C :: Vector{T}
    S :: Vector{T}
    tmpcosk :: Array{T, 2}
    tmpsink :: Array{T, 2}
    rkn :: symplectic_rkn4{T}
    dt :: T
    type

    function ParticleMover{T}(particles::Particles, meshx, K, dt; kx=1) where T<:Real
        Φ = Array{T, 2}(undef, 2, particles.nbpart)
        ∂Φ = similar(particles.x)
        tmpcosk = Array{T, 2}(undef, 1, particles.nbpart)
        tmpsink = Array{T, 2}(undef, 1, particles.nbpart)

        new(Φ,
            ∂Φ,
            meshx, meshx.stop, kx, K, 
            Vector{T}(undef, 2K+1), #C
            Vector{T}(undef, 2K+1), #S
            tmpcosk,
            tmpsink,
            symplectic_rkn4{T}(particles.x, dt), # rkn
            # rkn5{T}(particles.x, dt),
            dt, T)
    end
end



#==== Time steppers ====#
"""symplectic_RKN_order4!(X, V, F, rkn, kx)
    
    Advect (X, V) on a time step dt using symplectic Runge-Kutta-Nystrom method of order4 [Feng, Qin (2010), sect.7.3, p.327, scheme1].

    The equation satisfied by X is
    ```math
    \\frac{d^2 X(t)}{dt^2} = C(t)\\cos(X(t)) - S(t)\\sin(X(t))
    ```

    RKN method considers Ẋ = V as a variable, and updates both X and V.

    Args:
    - X: matrix of positions at time t_n
    - V: matrix of velocities at time t_n
    - F: values of initial condition at time t_0
    - rkn: rkn_order_4 struct, storing butcher tableau and pre-allocated arrays (holds the value of dt)
    - kx: 2π/L

    Updates X, V in place, and returns coefficients C, S at current time.

"""
function RKN_timestepper!(p, pmover, kernel)
    @views begin
        for s = 1:pmover.rkn.nb_steps
            @. pmover.rkn.G = p.x + p.v * pmover.rkn.c[s];

            for ss = 1:pmover.rkn.nb_steps
                @. pmover.rkn.G +=  pmover.rkn.a[s, ss] * pmover.rkn.fg[ss];
            end
            
            kernel(pmover.rkn.fg[s], pmover.rkn.G, p, pmover);
        end
    
        @. p.x += pmover.dt * p.v;
        for s=eachindex(pmover.rkn.b̄)
            @. p.x += pmover.rkn.b̄[s] * pmover.rkn.fg[s];
            @. p.v += pmover.rkn.b[s] * pmover.rkn.fg[s];
        end

        kernel(pmover.∂Φ, p.x, p, pmover);
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
function strang_splitting!(particles, pmover, kernel)  
    pmover.Φ .= 0
    pmover.∂Φ .= 0
    @. particles.x += particles.v * pmover.dt / 2
    kernel(pmover.∂Φ, particles.x, particles, pmover)
    @. particles.v += pmover.dt * pmover.∂Φ
    @. particles.x += particles.v * pmover.dt / 2
    kernel(pmover.∂Φ, particles.x, particles, pmover)
end


# ===== Kernel computations ==== #
"""
Compute -∂_x Φ[f](`x`) and stores it in `dst`. Also updates `pmover.Φ`
"""
function kernel_poisson!(dst, x, p, pmover)
    dst .= 0
    pmover.Φ .= 0

    pmover.C .= 0
    pmover.S .= 0

    @views for idxk=eachindex(pmover.C) 
        ξk = 2π .* (idxk .- (pmover.K+1)) ./ pmover.meshx.stop
        normξk² = sum(ξk.^2)
        (normξk² == 0 || sum(abs.(idxk .- (pmover.K+1))) > pmover.K) && continue

        for (idx, pos) = enumerate(eachcol(x))
            pmover.tmpsink[1, idx] = sin(dot(pos, ξk))
            pmover.tmpcosk[1, idx] = cos(dot(pos, ξk))
        end
        
        for i = 1:p.nbpart
            pmover.C[idxk] += pmover.tmpcosk[1, i] * p.β[i]
            pmover.S[idxk] += pmover.tmpsink[1, i] * p.β[i]
        end
        
        @. pmover.Φ -= (pmover.C[idxk] * pmover.tmpcosk + pmover.S[idxk] * pmover.tmpsink) / normξk²
        # The line below computes -∂Φ[f](`x`) and stores it to `dst`. 
        # Changing dynamics : 
        #   "+=": repulsive potential (plasmas dynamics)
        #   "-=": attractive potential (galaxies dynamics)
        @. dst += (pmover.C[idxk] * pmover.tmpsink - pmover.S[idxk] * pmover.tmpcosk) * ξk / normξk²
    end
    
    dst ./= pmover.torus_size
    pmover.Φ ./= pmover.torus_size
end

function kernel_gyrokinetic!(dst, x, p, pmover)
    dst .= .-x
end



# ===== Some quantities we can compute at each step ==== #
"""compute_electricalenergy²(p, pmover)

    Returns the square of the electrical energy
"""
function compute_electricalenergy²(pmover)
    elec_e² = 0.0
    for idxk = eachindex(pmover.C) 
        ξk = 2π .* (idxk .- (pmover.K+1)) ./ pmover.meshx.stop
        normξk² = sum(ξk.^2)
        (normξk² == 0 || sum(abs.(idxk .- (pmover.K+1))) > pmover.K)&&continue
        elec_e² += (pmover.C[idxk]^2 + pmover.S[idxk]^2) / normξk²
    end
    return elec_e² / pmover.torus_size
end


function compute_momentum(particles)
    mom = similar(@views particles.v[:, 1])
    for (idv, vv) = enumerate(eachcol(particles.v))
        mom .+= vv .* particles.β[idv]
    end
    return mom    
end

function compute_totalenergy²(particles, Eelec²)
    kin_e² = 0.0
    @views for (idv, vv) = enumerate(eachcol(particles.v))
        kin_e² += sum(abs2, vv) * particles.β[idv]

    end
    return 1/2 * (Eelec² + kin_e²)
end



""" periodic_boundary_conditions!(p, pmover)

    Impose periodic boundary conditions in space.
"""
function periodic_boundary_conditions!(p, pmover)
    @views for idx=1:p.nbpart
        for d=1:length(p.x[:, 1])
            (p.x[d, idx] > pmover.meshx.stop)&&(p.x[d, idx] -= pmover.meshx.stop)
            (p.x[d, idx] < 0)&&(p.x[d, idx] += pmover.meshx.stop) 
        end
    end
end



function WPM_step!(p::Particles, pmover::ParticleMover; kernel=kernel_poisson!)
    RKN_timestepper!(p, pmover, kernel)
    # strang_splitting!(p, pmover, kernel)
    # strang_splitting_implicit!(p, pmover, kernel)
    
    periodic_boundary_conditions!(p, pmover)
    
    E² = compute_electricalenergy²(pmover)
    mom = compute_momentum(p)
    Etot² = compute_totalenergy²(p, E²)
    return E², mom, Etot²
end