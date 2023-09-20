using Revise

using LinearAlgebra, QuadGK, Roots, FFTW, FastGaussQuadrature, SpecialFunctions, StaticArrays
using Triangulate
using VlasovSolvers
import VlasovSolvers: advection!

import VlasovSolvers: Particles, WPM_step!, ParticleMover, kernel_poisson!, kernel_poisson_nfft!, kernel_gyrokinetic!, kernel_freestreaming!, kernel_poisson_finufft!
using ProgressMeter, Printf
using Plots, LaTeXStrings

include("numerical_examples.jl")
include("pic.jl")




struct RectangleRule
    len::Int64
    start::Float64
    stop::Float64
    points::Vector{Float64}
    weights::Vector{Float64}
    step::Float64

    function RectangleRule(len, start, stop)
        points = LinRange(start, stop, len + 1)[1:end-1]
        s = step(points)
        weights = [s for _ = 1:len]
        new(len, start, stop, vec(points), weights, s)
    end
end


##################### TRIANGULATION #####################
function triangulation_interpolation(_valsf, _pos, _vel, gridxoutput, gridvoutput, example;
    returntriangulation=false)
    """
    Perform a linear interpolation of the data to the grid by using a Delaunay triangulation.

    Steps:
    1. Create the Delaunay triangulation
    2. For each point (x,v) in (gridxoutput, gridvoutput)
    a. Get the triangle T in which (x,v) lies, by iterating over all the triangles and testing each one of them
    b. From the vertices of T get a linear interpolation of the function at x, using barycentric coordinates
    """

    myoutput = similar(gridxoutput)

    pointset = Triangulate.TriangulateIO()

    # Copy some rightmost particles to the left, and some leftmost particles to the right. 
    # This allows to interpolate values from a periodic triangulation.
    κ = 1 / 20
    rightmost_particles = findall(>((1 - κ) * example.L[1]), _pos)
    leftmost_particles = findall(<(κ * example.L[1]), _pos)

    pos = append!(_pos[rightmost_particles] .- example.L[1], _pos, _pos[leftmost_particles] .+ example.L[1])
    vel = append!(_vel[rightmost_particles], _vel, _vel[leftmost_particles])
    valsf = append!(_valsf[rightmost_particles], _valsf, _valsf[leftmost_particles])

    @views pointset.pointlist = vcat(pos', vel')
    pointset.pointattributelist = valsf'
    (triangulation, _) = Triangulate.triangulate("Q", pointset)

    # Interpolate using the triangulation
    @views @inbounds for part = eachindex(gridxoutput)
        myoutput[part] = triangleContainingPoint(triangulation, gridxoutput[part], gridvoutput[part])[1]
        (myoutput[part] < 0) && (myoutput[part] = 0)
        # Usually happens in corners or at the top, so we can reasonably suppose the value if small enough to be 
        # approximated by zero.
    end

    returntriangulation ? nothing : triangulation = nothing
    return vec(myoutput), triangulation
end

function triangleContainingPoint(triangulation, x, v)
    @views @inbounds for (idxA, idxB, idxC) = eachcol(triangulation.trianglelist)

        A = triangulation.pointlist[:, idxA]
        B = triangulation.pointlist[:, idxB]
        C = triangulation.pointlist[:, idxC]

        # if x is outside of the rectangle defined by [minX(A, B, C), maxX(A, B, C)],
        # or v is outside of the rectangle defined by [minV(A, B, C), maxV(A, B, C)],
        # then we don't have to do the computations
        if ((x < A[1]) && (x < B[1]) && (x < C[1])) || ((x > A[1]) && (x > B[1]) && (x > C[1]))
            continue
        elseif ((v < A[2]) && (v < B[2]) && (v < C[2])) || ((v > A[2]) && (v > B[2]) && (v > C[2]))
            continue
        end

        # Use barycentric coordinates:
        det = (A[1] - C[1]) * (B[2] - C[2]) - (A[2] - C[2]) * (B[1] - C[1])
        λ₁ = (x - C[1]) * (B[2] - C[2]) + (v - C[2]) * (C[1] - B[1])
        λ₂ = (x - C[1]) * (C[2] - A[2]) + (v - C[2]) * (A[1] - C[1])
        λ₁ /= det
        λ₂ /= det

        if (λ₁ ≥ 0) && (λ₂ ≥ 0) && (λ₁ + λ₂ ≤ 1)
            wA = triangulation.pointattributelist[1, idxA]
            wB = triangulation.pointattributelist[1, idxB]
            wC = triangulation.pointattributelist[1, idxC]
            return wA * λ₁ + wB * λ₂ + (1 - λ₁ - λ₂) * wC, (idxA, idxB, idxC)
        end
    end

    return -1.0, (-1, -1, -1)
end


##################### WPM #####################
function solve_WPM!(nsteps, dt, particles, example, weights;
    plotting=false::Bool, kernel=kernel_poisson!, R=NaN, K=1)
    init_pos = copy(particles.x)
    init_vel = copy(particles.v)

    np = particles.nbpart

    results = (Eelec²=Array{Float64}(undef, nsteps),
        Etot²=Array{Float64}(undef, nsteps),
        momentum=Array{Float64,2}(undef, length(particles.v[:, 1]), nsteps),
        L²norm²=Array{Float64}(undef, nsteps),
        C=Array{Float64,length(example.dim) + 1}(undef, fill(2K + 1, example.dim)..., nsteps),
        S=Array{Float64,length(example.dim) + 1}(undef, fill(2K + 1, example.dim)..., nsteps))

    pmover = ParticleMover{particles.type,particles.dim}(particles, example.L, K, dt)

    if (plotting) && (example.dim == 1)
        widthx = example.L[1]
        widthv = 2example.vmax[1]
        scale = 1
        plotEelec = plot([], [], xlim=(0, nsteps * dt), yaxis=:log10)
    end

    progression = ProgressMeter.Progress(nsteps, desc="Loop in time (WPM): ", showspeed=true)
    animation = @animate for istep = 1:nsteps
        WPM_step!(particles, pmover; kernel=kernel)
        results.momentum[:, istep] = pmover.momentum
        (results.Eelec²[istep], results.Etot²[istep]) = pmover.Eelec²tot²

        results.S[fill(:, example.dim)..., istep] .= pmover.S
        results.C[fill(:, example.dim)..., istep] .= pmover.C

        # L² norm:
        results.L²norm²[istep] = 0.0
        for i = 1:particles.nbpart
            results.L²norm²[istep] += particles.β[i]^2 / vec(weights)[i]
        end

        if (istep % R == 0) && (example.dim == 1)
            particles.β .= triangulation_interpolation(particles.β ./ vec(weights), particles.x[1, :],
                particles.v[1, :],
                init_pos[1, :], init_vel[1, :], example)[1]
            particles.β .*= vec(weights)
            particles.x .= init_pos
            particles.v .= init_vel
        end

        if (plotting) && (example.dim == 1)
            plotParts = plot(particles.x[1, :], particles.v[1, :], seriestype=:scatter, zcolor=vec(particles.β),
                markersize=sqrt(0.1) * scale, zticks=[], camera=(0, 90),
                markerstrokecolor="white", markerstrokewidth=0, label="", c=:jet1,
                size=(600, 600),
                title="t = $(@sprintf("%.3f",istep*dt))\nProgression: $(round(Int64,100*progression.counter / progression.n))%", titlefontsize=8, margin=5Plots.mm)
            push!(plotEelec, istep * dt, sqrt(results.Eelec²[istep]))
            plot(plotParts, plotEelec, layout=@layout [a; b])
        end

        ProgressMeter.next!(progression)
    end when plotting
    if !plotting
        animation = nothing
    end

    return results, animation
end


######################


function run_WPM(example, T, dt, nstep, K, mytype, nxwpm, nvwpm, quadX, quadV)
    quadrulesx = [quadX(nxwpm, 0, example.L[d]) for d = 1:example.dim];
    quadrulesv = [quadV(nvwpm, example.vmin[d], example.vmax[d]) for d = 1:example.dim];

    R = NaN
    dim = example.dim


    ##################### PARTICLES (WPM) #####################

    nbparticles = *(fill(nxwpm, dim)...) * *(fill(nvwpm, dim)...)

    x0_init = map(x -> [xx for xx in x], vec(collect(Base.product([rulex.points for rulex = quadrulesx]...))))
    v0_init = map(v -> [vv for vv in v], vec(collect(Base.product([rulev.points for rulev = quadrulesv]...))))
    x = map(z -> z[1], vec(collect(Base.product(x0_init, v0_init))))
    v = map(z -> z[2], vec(collect(Base.product(x0_init, v0_init))))

    wx = map(prod, vec(collect(Base.product([rulex.weights for rulex = quadrulesx]...))))
    wv = map(prod, vec(collect(Base.product([rulev.weights for rulev = quadrulesv]...))))
    weights = map(prod, vec(collect(Base.product(wx, wv))))

    fvals = example.f0.(x, v)
    wei = fvals .* weights

    x0 = reduce(hcat, x)
    v0 = reduce(hcat, v)

    particles = Particles{mytype}(x0, v0, wei);

    resWPM, animWPM = solve_WPM!(nstep, dt, particles, example, weights;
        plotting=false, kernel=kernel_poisson_finufft!, R=R, K=K);
    # gif(animWPM)

    t = (1:nstep) .* dt
    ε = 1e-18



    # ##################### PLOTS #####################

    # ##### Plot particles as a scatter plot #####
    # # scale = 5
    # # plot(particles.x[1, :], particles.v[1, :], seriestype=:scatter, zcolor=vec(particles.β),
    # #     markersize=sqrt(0.1) * scale, zticks=[], camera=(0, 90),
    # #     markerstrokecolor="white", markerstrokewidth=0, label="", c=:jet1,
    # #     size=(600, 600),
    # #     title="t = $(T)", titlefontsize=8, margin=5Plots.mm)
    # # xlabel!("x")
    # # ylabel!("v")
    # ############################################



    # ##### Interpolate values using triangulation,
    # ##### then do a 3d plot
    # quadrulesx = [quadX(nxwpm, 0, example.L[d]) for d = 1:example.dim];
    # quadrulesv = [quadV(nvwpm, example.vmin[d], example.vmax[d]) for d = 1:example.dim];

    # xout = map(x -> [xx for xx in x], vec(collect(Base.product([rulex.points for rulex = quadrulesx]...))))
    # vout = map(v -> [vv for vv in v], vec(collect(Base.product([rulev.points for rulev = quadrulesv]...))))

    # xout = map(z -> z[1], xout)
    # vout = map(z -> z[1], vout)

    # xout0 = reduce(hcat, map(z -> z[1], vec(collect(Base.product(xout, vout)))))
    # vout0 = reduce(hcat, map(z -> z[2], vec(collect(Base.product(xout, vout)))))


    # vals_on_grid_vec = triangulation_interpolation(particles.β ./ vec(weights), 
    #     particles.x[1, :], particles.v[1, :], xout0[1, :], vout0[1, :], example)[1]
    # vals_on_grid = reshape(vals_on_grid_vec, size(xout)[1], size(vout)[1])


    # myplot = heatmap(xout, vout, vals_on_grid', right_margin=5Plots.mm)
    # xlabel!(myplot, "x")
    # ylabel!(myplot, "v")
    # title!(myplot, example.longname * ", t = $(T)")
    # IMG_DIR = "/Users/ylehenaf/Documents/confs_presentations/CJC-MA-2023/beamer/png/"
    # fn = IMG_DIR * "WPM--" * example.shortname * "--dt_$(dt)--T_$(T)--K_$(K)--$(quadX)$(nxwpm)-$(quadV)$(nvwpm)--vmax_$(example.vmax[1])--kx_$(example.kxs[1]).png"
    # # savefig(myplot, fn)
    # display(myplot)
    #############################################


    ###
    # REPRENDRE LE CALCUL DE E POUR PIC, ET OBTENIR LES BONNES VALEURS DÉJÀ EN t=0
    ###



    ###### 1D PIC simulations ######
    nbparticles_PIC = Int64(1e5)
    gridsize_PIC = 2^4

    pic_particles = sample_PIC_particles(nbparticles_PIC, example)
    # histogram2d(pic_particles.x[1, :], pic_particles.v[1, :], weights=pic_particles.β, bins=250)
    # plot(pic_particles.x[1, :], pic_particles.v[1, :], zcolor=pic_particles.β, seriestype=:scatter)
    ########
    elec, mom, tot = solve_PIC!(nstep, dt, pic_particles, example, gridsize_PIC,plotting=false)


    plotEelec = plot(minorgrid=true, size=(600, 400), yaxis=:log10, legend=:outerright)
    plot!(t .+ dt, max.(ε, sqrt.(resWPM.Eelec²)), label="WPM")
    plot!(t, max.(ε, sqrt.(elec)), label="PIC")
    title!(L"PIC: E_{elec}, v_{max}=" * "$(example.vmax)" * ", Np = $nbparticles_PIC")
    # plot(dt .* (1:nstep), mom, label="Mom PIC")
    # hline!([example.momentumexact])
    # plot(dt .* (1:nstep), tot, label="Etot PIC")
    # hline!([example.Etot²exact])
    #################################################








    # t = (1:nstep) .* dt

    # plotCk = plot(minorgrid=true, size=(600, 400), legend=:outerright)
    # for k = -K:K
    #     plot!(t, resWPM.C[k+K+1, :], label=L"k=" * "$k")
    # end
    # title!(L"C_k")

    # plotSk = plot(minorgrid=true, size=(600, 400), legend=:outerright)
    # for k = -K:K
    #     plot!(t, resWPM.S[k+K+1, :], label=L"k=" * "$k")
    # end
    # title!(L"S_k")

    # plotCk²Sk² = plot(minorgrid=true, size=(600, 400), yaxis=:log10, legend=:outerright)
    # ε = 0
    # for k = -K:K
    #     if sum(abs.(k)) == 0
    #         plot!([], [], label="")
    #     else
    #         plot!(t, max.(ε, resWPM.S[k+K+1, :] .^ 2 .+ resWPM.C[k+K+1, :] .^ 2), label=L"k=" * "$k")
    #     end
    # end
    # title!(L"\sqrt{S_k^2 + C_k^2}")

    # plotEelec = plot(minorgrid=true, size=(600, 400), yaxis=:log10, legend=:outerright)
    # plot!(t .+ dt, max.(ε, sqrt.(resWPM.Eelec²)), label=L"E_{elec}")
    # title!(L"E_{elec}, v_{max}=" * "$(example.vmax)")

    # plot(plotCk, plotSk, plotCk²Sk², plotEelec, size=(1200, 400), layout=@layout [a b; c d])

end


function main()
    ##################### INPUTS #####################

    ##################### Examples
    example = example_landaudamping_1D
    # example = example_stronglandaudamping_1D
    # example = example_landaudamping_2D
    # example = example_twostreaminstability_1D
    # example = example_bumpontail_1D
    # example = example_stationarygaussian_1D
    # example = example_nonhomogeneousstationarysolution_1D
    # example = example_test_1D
    #####################

    T = 20.0
    dt = 0.1
    nstep = convert(Int64, floor(T / dt))
    # 
    K = 3

    mytype = Float64

    quadX = RectangleRule
    quadV = RectangleRule
    nxwpm = 196
    nvwpm = 197

    run_WPM(example, T, dt, nstep, K, mytype, nxwpm, nvwpm, quadX, quadV)
end

main()


##################### BENCHMARKS #####################
## Threading for loop over columns of p.x
# 1threads : 11.479219 seconds (210.41 k allocations: 18.148 MiB)
# 2threads : mauvais résultats, plus long
# 4threads : mauvais résultats, plus long

## Threading for loop over idxk
# 1threads : 11.441745 seconds (1.18 M allocations: 64.114 MiB, 0.20% gc time, 3.60% compilation time)
# 2threads : mauvais résultats, plus long
# 4threads : mauvais résultats, plus long
