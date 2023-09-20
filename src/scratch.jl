# using Plots
# using NFFT, FINUFFT


# struct TrapezoidalRule
#     len::Int64
#     start::Float64
#     stop::Float64
#     points::Vector{Float64}
#     weights::Vector{Float64}
#     step::Float64

#     function TrapezoidalRule(len, start, stop)
#         points = LinRange(start, stop, len)[1:end]
#         s = step(points)
#         weights = [s for _ = 1:len]
#         weights[1] /= 2
#         weights[end] /= 2
#         new(len, start, stop, vec(points), weights, s)
#     end
# end


# # rho(x) = 2sin(x) + sin(2x) + 0.5sin(4x)
# # ∂²rho(x) = -2sin(x) - 4sin(2x) - 0.5 * 16sin(4x)

# kx = 0.5
# rho(x) = 2sin(kx*x) + sin(2kx*x) + 0.5sin(3kx*x)
# ∂rho(x) = 2kx * cos(kx*x) + 2kx * cos(2kx*x) + 0.5kx * 3cos(3kx*x)
# # ∂²rho(x) = -2sin(x) - 4sin(2x) - 0.5 * 16sin(4x)



# nbpts = 2^10
# L = 2π / kx
# quadrature = TrapezoidalRule(nbpts, 0, L)
# unifx = quadrature.points
# unifweights = quadrature.weights
# x = sort(rand(nbpts) .* L)
# weights = unifweights[:]
# y = convert.(ComplexF64, ∂rho.(x))
# # y = convert.(ComplexF64, ∂²rho.(x))


# # Number of required Fourier modes 
# # nbfouriermodes = 0.5 * nbpts;
# # nbfouriermodes = convert(Int64, 2 * ceil(nbfouriermodes / 2))
# nbfouriermodes = 8

# # required precision
# ε = 1e-12

# p = plan_nfft(x ./ L .- 0.5, nbfouriermodes) # create plan. m and σ are optional parameters
# nodes!(p, reshape(x ./ L .- 0.5, 1, :))
# fourier_transform = adjoint(p)
# inverse_fourier_transform = p
# ywithweights = y .* weights
# @time fourier_coeffs_∂rho = fourier_transform * (ywithweights) # Adjoint is Fourier transform

# @time fourier_coeffs_∂rho_finufft = nufft1d1(x ./ L .* 2π .- π,
#     ywithweights,
#     -1,
#     ε,
#     nbfouriermodes
# );


# true_fourier_coeffs_∂rho = nufft1d1(unifx ./ L .* 2π .- π,
#     ywithweights,
#     -1,
#     ε,
#     nbfouriermodes
# );

# println(sum(abs2.(fourier_coeffs_∂rho .- true_fourier_coeffs_∂rho)))
# println(sum(abs2.(fourier_coeffs_∂rho_finufft .- true_fourier_coeffs_∂rho)))


# println("Fourier coeffs")
# for k = 0:(nbfouriermodes-1)/2
#     idxk = Int64(k + nbfouriermodes / 2 + 1)
#     idxminusk = Int64(-k + nbfouriermodes / 2 + 1)
#     ck = 0.5 * (fourier_coeffs_∂rho[idxk] + fourier_coeffs_∂rho[idxminusk])
#     sk = 0.5 * (fourier_coeffs_∂rho[idxk] - fourier_coeffs_∂rho[idxminusk]) / 1im
#     trueck = 0.5 * (true_fourier_coeffs_∂rho[idxk] + true_fourier_coeffs_∂rho[idxminusk])
#     truesk = 0.5 * (true_fourier_coeffs_∂rho[idxk] - true_fourier_coeffs_∂rho[idxminusk]) / 1im
#     println("k=$k : \tck=$(real(ck))\ttrueck=$(real(trueck))\t|ck-trueck|^2 =$(abs2(ck-trueck)),\t|sk-truesk|^2 =$(abs2(sk-truesk))")
# end
# println("---")

# # # Prepare to divide by -k^2
# invks = collect(-nbfouriermodes/2:(nbfouriermodes-1)/2)
# invks[Int64(nbfouriermodes / 2 + 1)] = 1
# invks .*= 2π / L
# invks = 1 ./ invks
# invks[Int64(nbfouriermodes / 2 + 1)] = 0

# # fourierordered_invks = collect(0:(nbfouriermodes-1)/2)
# # append!(fourierordered_invks, -nbfouriermodes/2:-1)
# # fourierordered_invks[1] = 1
# # fourierordered_invks .= 1 ./ fourierordered_invks
# # fourierordered_invks[1] = 0
# # @show fourierordered_invks


# # # Divide fourier coefficients by k^2
# # @show size(fourier_coeffs_∂rho)
# # @show size(fourierordered_invks)
# fourier_coeffs_rho = fourier_coeffs_∂rho .* invks * 1im

# # display(fourier_coeffs_rho)

# # approx_rho = p * (fourier_coeffs_rho .* -1im)
# @time approx_rho = inverse_fourier_transform * fourier_coeffs_rho
# approx_rho ./= L


# # # Now perform inverse NUFFT to obtain approximate solution to Poisson equation
# @time approx_rho_finufft = nufft1d2(x ./ L .* 2π .- π,
#     1,
#     ε,
#     fourier_coeffs_∂rho_finufft .* invks .* -1im
# ) / L;

# # display(approx_rho)

# println(sum(abs2.(approx_rho .- rho.(x))))
# println(sum(abs2.(approx_rho_finufft .- rho.(x))))

# plot(x, real.(approx_rho), label="NFFT", ls=:dash)
# plot!(x, real.(approx_rho_finufft), label="FINUFFT", ls=:dot)
# # # plot(x, .-imag.(approx_rho), markers=:dot, markerstrokewidth=0, label="FINUFFT approx")
# plot!(truex -> real(rho(truex)), 0, L, label="exact")



# using NFFT

# M, N = 32, 2
# x = range(-0.4, stop=0.4, length=M)  # nodes at which the NFFT is evaluated
# fHat = randn(ComplexF64,M)           # data to be transformed
# p = plan_nfft(x, N)                  # create plan. m and σ are optional parameters
# f = adjoint(p) * fHat                # calculate adjoint NFFT
# g = p * f  



#############################################



# using FFTW
# using Plots

# include("pic.jl")

# kx = 0.5
# L = 2π / kx
# nx = 2^7
# mesh = range(0, L, nx+1)[1:end-1]
# dx = L / nx

# rho = sin.(kx * mesh)

# p1 = plot(mesh, rho)
# plot!(z -> sin(kx * z), seriestype=:scatter)
# display(p1)

# phi, E = compute_phi_E_from_rho(rho, dx, L)

# p2 = plot(mesh, phi)
# plot!(z -> - 1 / kx^2 * sin(kx * z), seriestype=:scatter)
# display(p2)

# p3 = plot(mesh, E)
# plot!(z -> -1 / kx * cos(kx * z), seriestype=:scatter)
# display(p3)


##########################################################


using Plots

include("WPM.jl")

include("pic.jl")
include("numerical_examples.jl")

function test_rho()

    # example = LandauDamping_ND(; alpha=0.01, kxs=[0.5], mu=[0.0], beta=[1.0], shortname="test_1D", longname="Test 1D", L=nothing, vmax=[12.0]);
    # example = example_landaudamping_1D
    # example = example_stronglandaudamping_1D
    example = example_twostreaminstability_1D
    nx = 2^5
    nbparts = 100_000

    L = example.L[1]
    mesh = range(0, L, nx+1)[1:end-1]
    dx = step(mesh)

    @show L
    particles = sample_PIC_particles(nbparts, example)
    phi, E, rho = compute_phi_E(particles, dx, nx, example, return_rho=true)
    @show sum(rho) * dx

    plotrho = plot(mesh, rho, label="rho approx")
    # plot!(particles.x[1, :], particles.β, seriestype=:scatter)
    xticks!(plotrho, (mesh, [" " for _ in mesh]))
    # Check that rho is consistent: since \int example.f0v(v) dv = 1, 
    # the approximate rho must be equal to example.f0x
    plot!(plotrho, mesh, example.f0x.(mesh), seriestype=:scatter, label="rho exact")


    # rho neutral is also consistent:
    rho_neutral = rho .- sum(rho) * dx / L
    plotrhoneutral = plot(mesh, rho_neutral, label="rho neutral approx")
    plot!(plotrhoneutral, mesh, example.α .* cos.(example.kxs[1] .* mesh), label="rho neutral exact", seriestype=:scatter)


    # phi consistent:
    plotphi = plot(mesh, phi, label="phi approx")
    plot!(plotphi, mesh, -example.α ./ example.kxs[1]^2 .* cos.(example.kxs[1] .* mesh)  ,label="phi exact", seriestype=:scatter)


    # E?
    plotE = plot(mesh, E, label="E approx")
    plot!(plotE, mesh, example.α ./ example.kxs[1] .* sin.(example.kxs[1] .* mesh)  ,label="E exact", seriestype=:scatter)
    
    
    plot(plotrho, plotrhoneutral, plotphi, plotE, layout=@layout([a; b; c; d]), size=(600, 600))
end

# test_rho()





##########################################################


# using Plots
# using ProgressMeter

# include("pic.jl")

# kx=  0.5
# L = 2π / kx
# nx = 10
# dx = L / nx

# N = 1
# x = rand(N) .* L
# wei = 1 .* ones(N)
# mesh = vec(range(0, L, nx+1)[1:end-1])

# rho  = compute_rho(x, wei, dx, nx)

# p1 = plot(x, wei, seriestype=:scatter)
# plot!(mesh, rho)
# xticks!(mesh)

# nsteps = 1000
# progression = ProgressMeter.Progress(nsteps, desc="Scratch: ", showspeed=true)
# animation = @animate for xx in vec(range(0, L, nsteps+1)[1:end-1])
#     x = [xx]
#     rho  = compute_rho(x, wei, dx, nx)
#     p1 = plot(x, wei, seriestype=:scatter)
#     plot!(mesh, rho)
#     xticks!(mesh)
#     ProgressMeter.next!(progression)
# end
# gif(animation)



##########################################################

using Plots
using Sobol
using ProgressMeter

include("WPM.jl")

include("pic.jl")
include("numerical_examples.jl")



"""Calcul de la 'inverse cumulative distribution'"""
function test_inverse_CDF()

    example = example_landaudamping_1D
    # example = example_stronglandaudamping_1D
    # example = example_twostreaminstability_1D
    nx = 2^6
    nbparts = 10_000
    
    particles = sample_PIC_particles(nbparts, example)

    plotscatter = scatter(particles.x[1, :], particles.v[1, :], zcolor=particles.β)
    display(plotscatter)
    
    histx = histogram(particles.x[1, :], normalize=true, bins=nx, linewidth=0.)
    ## Focus
    # y = histx[1][1][:y]
    # y .-= y[1]
    # extrema_y = extrema(x->isnan(x) ? y[1] : x, y[1:6:end][begin:end-1]) 
    # ylims!(histx, extrema_y)
    ########
    display(histx)
end

test_inverse_CDF()
