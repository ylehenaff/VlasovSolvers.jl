using FFTW
using StatsBase
using SparseArrays
using Printf
using Sobol

# 1D PIC method using a linear interpolation deposition step


function solve_PIC!(nsteps, dt, particles, example, gridsize; plotting=false)
    progression = ProgressMeter.Progress(nsteps, desc="Loop in time (PIC): ", showspeed=true)
    elec_history = zeros(Float64, nsteps)
    tot_history = zeros(Float64, nsteps)
    mom_history = zeros(Float64, nsteps)
    dx = step(LinRange(0, example.L[1], gridsize))
    plotEelec = plot([], [], xlim=(0, nsteps * dt), yaxis=:log10)
    #
    _, E = compute_phi_E(particles, dx, gridsize, example)
    elec_history[1] = sum(E.^2) * dx
    mom_history[1] = sum(particles.v[1, :] .* particles.β)
    tot_history[1] = (elec_history[1] + sum(particles.v[1, :].^2 .* particles.β)) / 2.0
    #
    animation = @animate for istep in 2:nsteps
        x_step!(particles, 0.5dt, dx, example)
        v_step!(particles, dt, gridsize, dx)
        x_step!(particles, 0.5dt, dx,example)
        _, E = compute_phi_E(particles, dx, gridsize, example)
        elec_history[istep] = sum(E.^2) * dx
        mom_history[istep] = sum(particles.v[1, :] .* particles.β)
        tot_history[istep] = (elec_history[istep] + sum(particles.v[1, :].^2 .* particles.β)) / 2.0

        ProgressMeter.next!(progression)

        if plotting
            plotParts = plot(particles.x[1, :], particles.v[1, :], seriestype=:scatter, 
                zcolor=vec(particles.β),
                markersize=sqrt(0.1) * 1, zticks=[], camera=(0, 90),
                markerstrokecolor="white", markerstrokewidth=0, label="", c=:jet1,
                size=(600, 600),
                title="t = $(@sprintf("%.3f",istep*dt))\nProgression: $(round(Int64,100*progression.counter / progression.n))%", titlefontsize=8, margin=5Plots.mm)
            push!(plotEelec, istep * dt, sqrt(elec_history[istep]))
            plot(plotParts, plotEelec, layout=@layout [a; b])
        end
    end when plotting
    if plotting
        gif(animation)
    end
    return elec_history, mom_history, tot_history
end


"""
Returns the largest index `idx` such that `array_approx_CDF[idx] < s`
"""
function inverse_CDF(array_approx_CDF, u::Float64)
    idx_highest_smaller = 1
    max_idx = length(array_approx_CDF)
    while (idx_highest_smaller < max_idx) && (array_approx_CDF[idx_highest_smaller] < u)
        idx_highest_smaller += 1
    end
    return idx_highest_smaller-1
end

"""
Same as inverse_CDF but perform it on a Vector of Float64 more efficiently.
"""
function inverse_CDF(array_approx_CDF, u::Vector{Float64})
    # get permutation needed to sort values 
    p = sortperm(u)
    # inverse permutation
    pinv = zeros(Int64, length(u))
    pinv[p] .= 1:length(u)
    # first sort values:
    usorted = u[p]
    idx_current_u = 1
    corresponding_indices = zeros(Int64, length(u))
    stop = false
    for idx = 1:length(array_approx_CDF)-1
        while !stop && (array_approx_CDF[idx] <= usorted[idx_current_u] < array_approx_CDF[idx+1])
            corresponding_indices[idx_current_u] =  idx
            idx_current_u += 1
            if idx_current_u > length(u)
                stop = true
            end
        end
        stop ? break : nothing
    end
    return corresponding_indices[pinv]
end


function sample_PIC_particles(nbparticles, example)
    # look for x such that \int_0^x example.fx(x) dx = r,
    # with r random.
    # This nonlinear equation is solved using the Newton method.
    #
    # Instead of finding x such that \int_0^x example.fx(x) dx = r,
    # compute an x-linspace and then fx(x-linspace).
    # then draw x according to the density fx
    print("Sampling PIC particles: ")

    # x sampling:
    #############
    unifgridsize = max(Int64(1e6), nbparticles)
    meshx = LinRange(0, example.L[1], unifgridsize+1)[1:end-1]
    x = meshx |> collect
    fx = example.f0x.(x)
    fx ./= sum(fx)
    cumulative_distribution_approx = cumsum(fx)
    cumulative_distribution_approx ./= cumulative_distribution_approx[end]

    low_discrepancy_seq = SobolSeq(1)
    skip(low_discrepancy_seq, nbparticles)
    # It is advised (https://github.com/JuliaMath/Sobol.jl) to skip the first
    # values to have better uniformity
    
    # xx = zeros(Float64, nbparticles)
    # progression = ProgressMeter.Progress(nbparticles, desc="Low-discrepancy sampling in x: ", showspeed=true)
    # for i=1:nbparticles
    #     s = Sobol.next!(low_discrepancy_seq)[1]
    #     idx_highest_smaller = inverse_CDF(cumulative_distribution_approx, s)
    #     xx[i] = meshx[idx_highest_smaller]
    #     ProgressMeter.next!(progression)
    # end
    sobolVals = [Sobol.next!(low_discrepancy_seq)[1] for _ in 1:nbparticles]
    indices = inverse_CDF(cumulative_distribution_approx, sobolVals)
    xx = x[indices]
    particles_x = reshape(xx, 1, nbparticles)

    
    # v sampling:
    #############
    meshv = LinRange(example.vmin[1], example.vmax[1], unifgridsize+1)[1:end-1]
    v = meshv |> collect
    fv = example.f0v.(v)
    fv ./= sum(fv)
    cumulative_distribution_approx = cumsum(fv)
    cumulative_distribution_approx ./= cumulative_distribution_approx[end]

    low_discrepancy_seq = SobolSeq(1)
    skip(low_discrepancy_seq, nbparticles)
    # It is advised (https://github.com/JuliaMath/Sobol.jl) to skip the first
    # values to have better uniformity

    # vv = zeros(Float64, nbparticles)
    # progression = ProgressMeter.Progress(nbparticles, desc="Low-discrepancy sampling in v: ", showspeed=true)
    # for i=1:nbparticles
    #     s = Sobol.next!(low_discrepancy_seq)[1]
    #     idx_highest_smaller = inverse_CDF(cumulative_distribution_approx, s)
    #     vv[i] = meshv[idx_highest_smaller]
    #     ProgressMeter.next!(progression)
    # end
    sobolVals = [Sobol.next!(low_discrepancy_seq)[1] for _ in 1:nbparticles]
    indices = inverse_CDF(cumulative_distribution_approx, sobolVals)
    vv = v[indices]
    particles_v = reshape(vv, 1, nbparticles)


    particles_pic = Particles{Float64}(particles_x, particles_v, (example.L[1] / nbparticles) .* ones(nbparticles))

    println("done.")
    enforce_x_periodicity!(particles_pic, example.L)
    return particles_pic
end


function x_step!(p, dt, dx, example)
    for i in eachindex(p.x)
        p.x[i] += p.v[i] * dt
    end
    # Make sure all the x values remain in the computational domain
    # using periodicity
    enforce_x_periodicity!(p, example.L)
end


function enforce_x_periodicity!(p, Ls)
    # Make sure all the x values remain in the computational domain
    # using periodicity
    for d in eachindex(Ls)
        for i in 1:p.nbpart
            if p.x[d, i] >= Ls[d]
                p.x[d, i] -= Ls[d]
                # println("one out")
            elseif p.x[d, i] < 0.
                p.x[d, i] += Ls[d]
                # println("one in")
            end
        end
    end
end


function v_step!(p, dt, gridsize, dx)
    _, E = compute_phi_E(p, dx, gridsize, example)
    for ipart in eachindex(p.x)
        igrid = Int64(fld(p.x[ipart], dx)) + 1
        t = (p.x[ipart] - (igrid-1)*dx) / dx

        p.v[ipart] += dt * (E[igrid] * (1-t) + E[igrid < gridsize ? igrid+1 : 1] * t)
    end
end


function compute_phi_E(p, dx, gridsize, example; return_rho=false)
    # compute rho in 1D so E can be obtained as Ê(k) = 1/ik ρ̂[k] in fourier
    # @show example.L[1]
    rho = compute_rho(p.x[1, :], p.β, dx, gridsize)
    if return_rho
        return compute_phi_E_from_rho(rho, dx, example.L[1])..., rho
    else
        return compute_phi_E_from_rho(rho, dx, example.L[1])
    end
end


function compute_rho(x, wei, dx, gridsize)
    rho = zeros(Float64, gridsize)
    for ipart in eachindex(x)
        # find in which cell the particle is located, then interpolate its mass
        # linearly between the two cell ends.
        igrid = Int64(fld(x[ipart], dx)) + 1
        t = (x[ipart] - (igrid-1)*dx) / dx
        rho[igrid] += wei[ipart] * (1-t)
        rho[igrid < gridsize ? igrid+1 : 1] += wei[ipart] * t
    end

    return rho / dx
end


function compute_phi_E_from_rho(rho, dx, L)
    ρ̂ = fft(rho .- sum(rho) * dx / L)
    ξ = Vector{Float64}(fftfreq(length(rho), 2π/dx))
    ξ[1] = 1
    ρ̂ ./= - ξ.^2
    ρ̂[1] = 0.
    ξ[1] = 0.
    phi = real.(ifft(ρ̂))
    E = ifft(ρ̂ .* 1im.*ξ)

    return phi, real.(E)
end

