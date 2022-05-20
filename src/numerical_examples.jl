struct LandauDamping_1D
    α
    kx
    μ
    β
    f0
    shortname
    longname
    L 
    vmin 
    vmax
    Etot²exact
    momentumexact
    L²norm²exact
    dim

    function LandauDamping_1D(; alpha=0.001, kx=0.5, mu=0.0, beta=1.0,
                            shortname="Landau", longname="(Strong/Weak) Landau damping", L=nothing, vmin=-9., vmax=9.)
        isnothing(L) ? L = 2π / kx : nothing
        f(x,v) = (1 + alpha * cos(kx*x)) * exp(- beta * (v-mu)^2 / 2) / √(2π/beta)
        new(alpha, kx, mu, beta, 
            f, 
            shortname, longname, 
            L, vmin, vmax, 
            0.5 * (L * (1+mu^2) + alpha^2 * L^3 / (8π^2)), # Etot
            L*mu, # momentum
            0.5 * L * (1 + alpha^2/2) * sqrt(beta/π), # (L² norm)^2
            1
        )
    end
end

# ==================

struct TwoStreamInstability_1D 
    α
    kx
    β
    f0
    shortname
    longname
    L 
    vmin 
    vmax
    v0
    Etot²exact
    momentumexact
    L²norm²exact
    dim

    function TwoStreamInstability_1D(; alpha=0.001, kx=0.2, v0=3.0, beta=1.0,
                                    shortname="TSI", longname="Two-Stream Instability", L=nothing, vmin=-9., vmax=9.)
        isnothing(L) ? L = 2π / kx : nothing
        f(x,v) = (1 + alpha * cos(kx*x)) * (exp(- beta*(v-v0)^2 / 2) + exp(- beta*(v+v0)^2 / 2)) / (2*√(2π/beta))
        new(alpha, kx, beta, 
            f, 
            shortname, longname, 
            L, 
            vmin, vmax, v0, 
            0.5 * (L * (1+v0^2) + alpha^2 * L^3 / (8π^2)), # Etot
            0., # momentum
            0.25 * L * (1 + alpha^2/2) * sqrt(beta/π) * (1 + exp(-beta * v0^2)), # (L² norm)^2
            1
        )
    end
end

struct TwoStreamInstabilityAlternativeFormulation_1D
    α
    kx
    β
    f0
    shortname
    longname
    L 
    vmin 
    vmax
    Etot²exact
    momentumexact
    L²norm²exact
    dim

    function TwoStreamInstabilityAlternativeFormulation_1D(; alpha=0.05, kx=0.2, beta=1.0,
                                    shortname="TSI_alt", longname="Two-Stream Instability Alternative Formulation", L=nothing, vmin=-9., vmax=9.)
        isnothing(L) ? L = 2π / kx : nothing
        f(x,v) = (1 - alpha * cos(kx*x)) * exp(- beta*v^2 / 2) * v^2 / √(2π/beta)
        new(alpha, kx, beta, 
            f, 
            shortname, longname, 
            L, 
            vmin, vmax,
            0.5 * (L * 3/beta^2 + alpha^2 * L^3 / (8π^2) / beta^2), # Etot
            0., # momentum
            3/8 * L * (1 + alpha^2/2) / sqrt(π*beta^3), # (L² norm)^2
            1
            )
    end
end

# ==================

struct BumpOnTail_1D
    α
    kx
    μ₁
    μ₂
    β₁
    β₂
    n₁
    n₂
    f0
    shortname
    longname
    L 
    vmin
    vmax
    Etot²exact
    momentumexact
    L²norm²exact
    dim
    

    function BumpOnTail_1D(; alpha=0.04, kx=0.3, mu1=0.0, mu2=4.5, beta1=1.0, beta2=4.0,
                        n1=0.9, n2=0.2, shortname="BoT", longname="Bump on Tail", L=nothing, vmin=-9., vmax=9.)
        isnothing(L) ? L = 2π / kx : nothing
        f(x,v) = (1 + alpha * cos(kx*x)) * 
            (n1*exp(-beta1*(v-mu1)^2 / 2) + n2*exp(-beta2*(v-mu2)^2 / 2)) / (n1*sqrt(2π/beta1) + n2*sqrt(2π/beta2))
        new(alpha, kx, 
            mu1, mu2, 
            beta1, beta2, 
            n1, n2, 
            f, 
            shortname, longname, 
            L, 
            vmin, vmax, 
            0.5 * L * ((
                        n1 * sqrt(2π/beta1) * (1 / beta1 + mu1^2)+ 
                        n2 * sqrt(2π/beta2) * (1 / beta2 + mu2^2) 
                    ) / (n1*sqrt(2π/beta1) + n2*sqrt(2π/beta2)) + 
                    alpha^2 * L^2 / (8π^2)), # Etot
            L / (n1*sqrt(2π/beta1) + n2*sqrt(2π/beta2)) * (n1*mu1*sqrt(2π/beta1) + n2*mu2*sqrt(2π/beta2)), # momentum
            L * (1+alpha^2/2) / (2π) / (n1/sqrt(beta1) + n2/sqrt(beta2))^2 * (
                n1^2*sqrt(π/beta1) + n2^2*sqrt(π/beta2) + 
                    2*n1*n2*exp(-0.5( beta1*mu1^2 + beta2*mu2^2 - ( (beta1*mu1 + beta2*mu2) / sqrt(beta1+beta2) )^2 )) *
                    sqrt(2π / (beta1+beta2))
            ),
            1 # (L² norm)^2
        )
    end
end

# ==================

struct NonHomogeneousStationarySolution_1D
    α
    kx
    β
    M₀
    f0
    shortname
    longname
    L
    vmin
    vmax
    dim

    function getM₀(α, β)
        find_zero( (M) -> M - α * √(2π/β) * besseli(1, M * β) * 2, 10)
        # 2 factor because of the definition of I₁(z) and C(t):
        # I₁(z) = 1/π ∫_0^π exp(z cos(θ)) cos(θ) dθ
        #       = 1/(2π) ∫_0^{2π} exp(z cos(θ)) cos(θ) dθ
        # C(t)  = 1/π ∫_0^{2π} ∫_{-∞}^{+∞} f(t,θ,v) cos(θ) dθ dv
        #       = 2α √(2π/β) I₁(βM₀)
    end

    function NonHomogeneousStationarySolution_1D(; alpha=0.2, kx=1, beta=2,
                                                shortname="non-homog", longname="Non Homogeneous Stationary Solution", L=nothing, 
                                                vmin=-9., vmax=9.)
        isnothing(L) ? L = 2π / kx : nothing
        m = getM₀(alpha, beta)
        new(alpha, kx, beta, m, (x,v) -> alpha * exp.(-beta * (v^2 / 2 - m * cos(x*kx))), shortname, longname, L, vmin, vmax, 1)
    end
end

# ==================

struct StationaryGaussian_1D
    α
    kx
    β
    f0
    shortname
    longname
    L
    vmin
    vmax
    Etot²exact
    momentumexact
    L²norm²exact
    dim

    function StationaryGaussian_1D(; alpha=0.2, kx=1, beta=1,
                                                shortname="gaussian", longname="Stationary Gaussian", L=nothing, 
                                                vmin=-9., vmax=9.)
        isnothing(L) ? L = 2π / kx : nothing
        new(alpha, kx, beta, 
            (x,v) -> alpha * exp.(-beta * v^2 / 2) / √(2π/beta), 
            shortname, longname, 
            L, 
            vmin, vmax,
            0.5 * alpha * 1/beta * L, # Etot
            0., # momentum
            L * alpha^2/2 * sqrt(beta/π), # (L² norm)^2
            1
        )
    end
end

# ==================

struct Test_1D
    α
    kx
    β
    f0
    shortname
    longname
    L
    vmin
    vmax
    dim

    function Test_1D(; alpha=0.2, kx=1, beta=1,
                                shortname="test", longname="Test", L=nothing, 
                                vmin=-9., vmax=9.)
        isnothing(L) ? L = 2π / kx : nothing
        new(alpha, kx, beta, (x,v) -> alpha * exp.(-beta * (v - (vmin+vmax)/2)^2 / 2) / √(2π/beta) * exp.(-beta * (x - L/2)^2 / 2) / √(2π/beta), shortname, longname, L, vmin, vmax, 1)
    end
end

# ==================

example_landaudamping_1D = LandauDamping_1D(alpha=0.001, kx=0.5, mu=0., beta=1.,
                                        longname="Weak Landau damping", shortname="weakLD", 
                                        vmin=-12., vmax=12.);
example_stronglandaudamping_1D = LandauDamping_1D(alpha=0.5, kx=0.5, mu=0., beta=1., 
                                        longname="Strong Landau damping", shortname="strongLD", 
                                        vmin=-12., vmax=12.);
example_twostreaminstability_1D = TwoStreamInstability_1D(alpha=0.001, kx=0.2, v0=3., 
                                                    vmin=-12., vmax=12.);
example_twostreaminstabilityalternativeformulation_1D = TwoStreamInstabilityAlternativeFormulation_1D(alpha=0.05, kx=0.2,
                                                                                                vmin=-12., vmax=12.);
example_bumpontail_1D = BumpOnTail_1D(alpha=0.04, kx=0.3, mu1=0., mu2=4.5, beta1=1, beta2=4, 
                                vmin=-12., vmax=12.);
example_nonhomogeneousstationarysolution_1D = NonHomogeneousStationarySolution_1D(alpha=0.2, kx=1, beta=2, 
                                                                            vmin=-12., vmax=12.);
example_stationarygaussian_1D = StationaryGaussian_1D(alpha=0.2, kx=1, beta=1, 
                                                vmin=-12., vmax=12.);
example_test_1D = Test_1D(alpha=0.2, kx=1, beta=1, 
                    vmin=-12., vmax=12.);