using FFTW
using StatsBase
using SparseArrays


# 1D PIC method using a linear interpolation deposition step


function solve_PIC!(nsteps, dt, particles, example, gridsize)
    progression = ProgressMeter.Progress(nsteps, desc="Loop in time (PIC): ", showspeed=true)
    elec_history = zeros(Float64, nsteps)
    tot_history = zeros(Float64, nsteps)
    mom_history = zeros(Float64, nsteps)
    dx = example.L[1] / gridsize
    #
    _, E = compute_phi_E(particles, dx, gridsize, example)
    elec_history[1] = sum(E.^2) * dx
    mom_history[1] = sum(particles.v.^2 * particles.β)
    tot_history[1] = (elec_history[1] + mom_history[1]) / 2.0
    #
    for istep in 2:nsteps
        x_step!(particles, 0.5dt, example)
        v_step!(particles, dt, gridsize, dx)
        x_step!(particles, 0.5dt, example)
        _, E = compute_phi_E(particles, dx, gridsize, example)
        elec_history[istep] = sum(E.^2) * dx
        mom_history[istep] = sum(particles.v.^2 * particles.β)
        tot_history[istep] = (elec_history[istep] + mom_history[istep]) / 2.0

        ProgressMeter.next!(progression)
    end
    return elec_history, mom_history, tot_history
end


function sample_PIC_particles(nbparticles, example)
    # look for x such that \int_0^x example.fx(x) dx = r,
    # with r random.
    # This nonlinear equation is solved using the Newton method.
    #
    # Instead of finding x such that \int_0^x example.fx(x) dx = r,
    # compute an x-linspace and then fx(x-linspace).
    # then draw x according to the density fx
    print("Sampling PIC particles")

    x = range(0, example.L[1], 100 * nbparticles) |> collect
    fx = example.f0x.(x)
    fx ./= sum(fx)
    xx = sample(x, Weights(fx), nbparticles)
    particles_x = reshape(xx, 1, nbparticles)

    print(" -- sampled x")

    v = range(example.vmin[1], example.vmax[1], 100 * nbparticles) |> collect
    fv = example.f0v.(v)
    fv ./= sum(fv)
    vv = sample(v, Weights(fv), nbparticles)
    particles_v = reshape(vv, 1, nbparticles)

    println(" -- sampled v.")

    return Particles{Float64}(particles_x, particles_v, (example.L[1] / nbparticles) .* ones(nbparticles))
end


function x_step!(p, dt, example)
    for i in 1:p.nbpart
        p.x[i] -= p.v[i] * dt
        # Make sure all the x values remain in the computational domain
        # using periodicity
        if p.x[i] > example.L[1]
            p.x[i] -= example.L[1]
        elseif p.x[i] < 0
            p.x[i] += example.L[1]
        end
    end
end

function v_step!(p, dt, gridsize, dx)
    phi, E = compute_phi_E(p, dx, gridsize, example)
    for ipart in 1:p.nbpart
        igrid = Int64(fld(p.x[ipart], dx)) + 1
        t = (p.x[ipart] - (igrid-1)*dx) / dx

        p.v[ipart] -= dt * (E[igrid] * (1-t) + E[igrid < gridsize ? igrid+1 : 1] * t)
    end
end


function compute_phi_E(p, dx, gridsize, example)
    # compute rho in 1D so E can be obtained as Ê(k) = 1/ik ρ̂[k] in fourier
    rho = zeros(Float64, gridsize)
    for ipart in 1:p.nbpart
        # find in which cell the particle is located, then interpolate its mass
        # linearly between the two cell ends.
        igrid = Int64(fld(p.x[ipart], dx)) + 1
        t = (p.x[ipart] - (igrid-1)*dx) / dx
        rho[igrid] += p.β[ipart] * (1-t)
        rho[igrid < gridsize ? igrid+1 : 1] += p.β[ipart] * t
    end
    return compute_phi_E_from_rho(rho, dx, example.L[1])
end


function compute_phi_E_from_rho(rho, dx, L)
    ρ̂ = fft(rho .- sum(rho) * dx / L)
    ξ = Vector{Float64}(fftfreq(length(rho), 2π/dx))
    ξ[1] = 1
    ρ̂ ./= - ξ.^2
    ρ̂[1] = 0.
    ξ[1] = 0.
    phi = real.(ifft(ρ̂))
    E = real.(ifft(ρ̂ .* 1im.*ξ))
    return phi, E
end
