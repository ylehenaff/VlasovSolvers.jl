using FFTW, LinearAlgebra, Statistics

@testset "Vlasov-Ampere Landau 1D1V" begin

dev = CPU()                  # device
nx, nv = 64, 64              # grid resolution
stepper = StrangSplitting()  # timestepper
dt = 0.01                    # timestep
nsteps = 10                  # total number of time-steps

xmin, xmax = 0, 4π           # X Domain length (m)
vmin, vmax = -6, 6           # V Domain length (m)
α  = 0.5                     # Perturbation amplitude
kx = 0.5                     # Wave number of perturbation

xgrid = OneDGrid(dev, nx, xmin, xmax)
vgrid = OneDGrid(dev, nv, vmin, vmax)

f = DistributionFunction( xgrid, vgrid )

landau!(f, α, kx)

prob = VlasovProblem(f, Fourier(xgrid, vgrid), dev)

sol = solve!(prob, stepper, dt, nsteps )

@test true

end


