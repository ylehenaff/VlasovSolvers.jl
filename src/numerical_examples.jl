struct LandauDamping_ND
    α
    kxs
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

    function LandauDamping_ND(; alpha=0.001, kxs=[0.5], mu=[0.0], beta=[1.0],
        shortname="Landau2D", longname="(Strong/Weak) Landau damping 2D", L=nothing, vmax=[9.0])
        isnothing(L) ? L = 2π ./ kxs : nothing
        d = length(kxs)
        f(x, v) = (1 + alpha * *(cos.(kxs .* x)...)) * *([exp(-beta[dd] * (v[dd] - mu[dd])^2 / 2) / √(2π / beta[dd]) for dd = 1:d]...)
        Etot = -1.0
        mom = -1.0
        L²norm² = -1.0
        if d == 1
            Etot = 0.5 * (L[1] * (1 + mu[1]^2) + alpha[1]^2 * L[1]^3 / (8π^2))
            mom = L[1] * mu[1]
            L²norm² = 0.5 * L[1] * (1 + alpha^2 / 2) * sqrt(beta[1] / π)
        elseif d == 2
            nothing
        elseif d == 3
            nothing
        end
        new(alpha, kxs, mu, beta,
            f,
            shortname, longname,
            L, .-vmax, vmax,
            Etot,
            mom,
            L²norm²,
            d
        )
    end
end

# ==================

struct TwoStreamInstability_ND
    α
    kxs
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

    function TwoStreamInstability_ND(; alpha=0.001, kxs=[0.2], v0=[3.0], beta=[1.0],
        shortname="TSI", longname="Two-Stream Instability", L=nothing, vmax=[9.0])
        isnothing(L) ? L = 2π ./ kxs : nothing
        d = length(kxs)
        f(x, v) = (1 + alpha * *(cos.(kxs .* x)...)) * *([(exp(-beta[dd] * (v[dd] - v0[dd])^2 / 2) + exp(-beta[dd] * (v[dd] + v0[dd])^2 / 2)) / (2 * √(2π / beta[dd])) for dd = 1:d]...)
        if d == 1
            Etot = 0.5 * (L[1] * (1 + v0[1]^2) + alpha^2 * L[1]^3 / (8π^2))
            mom = 0.0
            L²norm² = 0.25 * L[1] * (1 + alpha^2 / 2) * sqrt(beta[1] / π) * (1 + exp(-beta[1] * v0[1]^2))
        elseif d == 2
            nothing
        elseif d == 3
            nothing
        end
        new(alpha, kxs, beta,
            f,
            shortname, longname,
            L,
            .-vmax, vmax, v0,
            Etot,
            mom,
            L²norm²,
            d
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
        shortname="TSI_alt", longname="Two-Stream Instability Alternative Formulation", L=nothing, vmin=-9.0, vmax=9.0)
        isnothing(L) ? L = 2π / kx : nothing
        f(x, v) = (1 - alpha * cos(kx * x)) * exp(-beta * v^2 / 2) * v^2 / √(2π / beta)
        new(alpha, kx, beta,
            f,
            shortname, longname,
            L,
            vmin, vmax,
            0.5 * (L * 3 / beta^2 + alpha^2 * L^3 / (8π^2) / beta^2), # Etot
            0.0, # momentum
            3 / 8 * L * (1 + alpha^2 / 2) / sqrt(π * beta^3), # (L² norm)^2
            1
        )
    end
end

# ==================

struct BumpOnTail_ND
    α
    kxs
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


    function BumpOnTail_ND(; alpha=0.04, kxs=[0.3], mu1=[0.0], mu2=[4.5], beta1=[1.0], beta2=[4.0],
        n1=0.9, n2=0.2, shortname="BoT", longname="Bump on Tail", L=nothing, vmax=[9.0])
        isnothing(L) ? L = 2π ./ kxs : nothing
        d = length(kxs)
        f(x, v) = (1 + alpha * *(cos.(kxs .* x)...)) *
                  *([(n1 * exp(-beta1[dd] * (v[dd] - mu1[dd])^2 / 2) + n2 * exp(-beta2[dd] * (v[dd] - mu2[dd])^2 / 2)) / (n1 * sqrt(2π / beta1[dd]) + n2 * sqrt(2π / beta2[dd])) for dd = 1:d]...)
        if d == 1
            Etot = 0.5 * L[1] * ((
                       n1 * sqrt(2π / beta1[1]) * (1 / beta1[1] + mu1[1]^2) +
                       n2 * sqrt(2π / beta2[1]) * (1 / beta2[1] + mu2[1]^2)
                   ) / (n1 * sqrt(2π / beta1[1]) + n2 * sqrt(2π / beta2[1])) +
                                 alpha^2 * L[1]^2 / (8π^2))
            mom = L[1] / (n1 * sqrt(2π / beta1[1]) + n2 * sqrt(2π / beta2[1])) * (n1 * mu1[1] * sqrt(2π / beta1[1]) + n2 * mu2[1] * sqrt(2π / beta2[1]))
            L²norm² = L[1] * (1 + alpha^2 / 2) / (2π) / (n1 / sqrt(beta1[1]) + n2 / sqrt(beta2[1]))^2 * (
                n1^2 * sqrt(π / beta1[1]) + n2^2 * sqrt(π / beta2[1]) +
                2 * n1 * n2 * exp(-0.5(beta1[1] * mu1[1]^2 + beta2[1] * mu2[1]^2 - ((beta1[1] * mu1[1] + beta2[1] * mu2[1]) / sqrt(beta1[1] + beta2[1]))^2)) * sqrt(2π / (beta1[1] + beta2[1]))
            )
        elseif d == 2
            nothing
        elseif d == 3
            nothing
        end
        new(alpha, kxs,
            mu1, mu2,
            beta1, beta2,
            n1, n2,
            f,
            shortname, longname,
            L,
            .-vmax, vmax,
            Etot,
            mom, # momentum
            L²norm²,
            d # (L² norm)^2
        )
    end
end

# ==================

struct NonHomogeneousStationarySolution_1D
    α
    kxs
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
        find_zero((M) -> M - 2α * √(2π / β) * besseli(1, M * β), +10)
        # 2 factor because of the definition of I₁(z) and C(t):
        # I₁(z) = 1/π ∫_0^π exp(z cos(θ)) cos(θ) dθ
        #       = 1/(2π) ∫_0^{2π} exp(z cos(θ)) cos(θ) dθ
        # C(t)  = 1/π ∫_0^{2π} ∫_{-∞}^{+∞} f(t,θ,v) cos(θ) dθ dv
        #       = 2α √(2π/β) I₁(βM₀)
    end

    function NonHomogeneousStationarySolution_1D(; alpha=0.2, kxs=[1.0], beta=[2.0],
        shortname="non-homog", longname="Non Homogeneous Stationary Solution", L=nothing,
        vmax=[9.0])
        isnothing(L) ? L = 2π ./ kxs : nothing
        m = getM₀(alpha, beta[1])
        new(alpha, kxs, beta, m, (x, v) -> alpha * exp.(-beta[1] * (v[1]^2 / 2 - m * sin(x[1] * kxs[1]))),
            shortname, longname, L, .-vmax, vmax, 1)
    end
end

# ==================

struct StationaryGaussian_1D
    α
    kxs
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

    function StationaryGaussian_1D(; alpha=0.2, kxs=[1.], beta=[1.],
        shortname="gaussian", longname="Stationary Gaussian", L=nothing,
        vmax=[9.0])
        isnothing(L) ? L = 2π ./ kxs : nothing
        d = length(kxs)
        f(x, v) = alpha * *([exp(-beta[dd] * v[dd]^2 / 2) / √(2π / beta[dd]) for dd = 1:d]...)
        Etot = -1.0
        mom = -1.0
        L²norm² = -1.0
        if d == 1
            Etot = 0.5 * alpha * 1 / beta[1] * L[1]
            mom = 0.0
            L²norm² = L[1] * alpha^2 / 2 * sqrt(beta[1] / π)
        elseif d == 2
            nothing
        elseif d == 3
            nothing
        end
        new(alpha, kxs, beta,
            f,
            shortname, longname,
            L,
            .-vmax, vmax,
            Etot, 
            mom, 
            L²norm²,
            d
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
        vmin=-9.0, vmax=9.0)
        isnothing(L) ? L = 2π / kx : nothing
        new(alpha, kx, beta, (x, v) -> alpha * exp.(-beta * (v - (vmin + vmax) / 2)^2 / 2) / √(2π / beta) * exp.(-beta * (x - L / 2)^2 / 2) / √(2π / beta), shortname, longname, L, vmin, vmax, 1)
    end
end

# ==================


## Landau damping
example_landaudamping_1D = LandauDamping_ND(; alpha=0.001, kxs=[0.5], mu=[0.0], beta=[1.0],
    shortname="weakLD_1D", longname="Weak Landau damping 1D", L=nothing, vmax=[12.0]);
example_stronglandaudamping_1D = LandauDamping_ND(alpha=0.5, kxs=[0.5], mu=[0.0], beta=[1.0],
    longname="Strong Landau damping 1D", shortname="strongLD_1D", vmax=[12.0]);
example_landaudamping_2D = LandauDamping_ND(; alpha=0.05, kxs=[0.5, 0.5], mu=[0.0, 0.0], beta=[1.0, 1.0],
    shortname="Landau_2D", longname="Weak Landau damping 2D", L=nothing, vmax=[12.0, 12.0]);

## TSI
example_twostreaminstability_1D = TwoStreamInstability_ND(alpha=0.001, kxs=[0.2], v0=[3.0], vmax=[12.0]);
example_twostreaminstabilityalternativeformulation_1D = TwoStreamInstabilityAlternativeFormulation_1D(alpha=0.05,
    kx=0.2, vmin=-12.0, vmax=12.0);

## Bump on Tail
example_bumpontail_1D = BumpOnTail_ND(alpha=0.04, kxs=[0.3], mu1=[0.0], mu2=[4.5], beta1=[1], beta2=[4],
    vmax=[12.0]);

## Nonhomogenous Stationary Solution
example_nonhomogeneousstationarysolution_1D = NonHomogeneousStationarySolution_1D(alpha=0.2, kxs=[1.0], beta=[2.0],
    vmax=[12.0]);

## Stationary gaussian
example_stationarygaussian_1D = StationaryGaussian_1D(alpha=0.2, kxs=[1.], beta=[1.],
    vmax=[12.0]);

## Others
example_test_1D = Test_1D(alpha=0.2, kx=1, beta=1,
    vmin=-12.0, vmax=12.0);