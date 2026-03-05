#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Oct 22 22:38:57 2025

@author: wyan0065
'''

import os
import sys
import h5py
import numpy as np
import xarray as xr
import irksome

import firedrake
from firedrake import inner, grad, dx, exp, min_value, max_value, Constant, derivative

sys.path.insert(0, '/g/data/rd53/wy2165/envs/icepack/icepack2-master')
import icepack
from icepack2.model import minimization, mass_balance
from icepack2.constants import gravity as g, ice_density as ρ_I, glen_flow_law as n, water_density as ρ_W

outdir = '/scratch/rd53/wy2165/heard_island/marine_terminating/'

U_c = Constant(100) # hybrid 100; SIA 0; SSA 100
#U_c = Constant(0)
#Λ = 1 # hybrid 1; SIA 0.00001; SSA 6.2956
#γ = 1 # hybrid 1; SIA 1; SSA 0
γ = 0
ela = Constant(243) # 534
c = Constant(0.4)

#U_cs = [100, 10, 1, 0.1, 0]
#Λs   = [1e1, 1e-1, 1e-2, 1e-3]
Λs   = [1e-3]
#γs   = [1e1, 1e-1, 1e-3, 1e-5]
#γs   = [0]
#elas = np.arange(534 - 500, 534, 50)
#elas = np.arange(534, 534 + 500 + 50, 50)
#cs = np.arange(0.15, 0.6+0.05, 0.05)
#cs = [0.004,0.04,4,40]

#for U_c in U_cs:
#    U_c = Constant(U_c);

for Λ in Λs:    
    Λ = Λ;

#for γ in γs:
#    γ = γ;

#for ela in elas:
#    ela = Constant(ela);

#for c in cs:
#    c = Constant(c);
    
    ############# mesh grid
    radius = Constant(14e3)
    mesh_disk = 6
    mesh = firedrake.UnitDiskMesh(mesh_disk)
    mesh.coordinates.dat.data[:] *= float(radius)

    cg = firedrake.FiniteElement('CG', 'triangle', 1)
    S = firedrake.FunctionSpace(mesh, cg)
    dg = firedrake.FiniteElement('DG', 'triangle', 0)
    Q = firedrake.FunctionSpace(mesh, dg)

    ############# geometry
    x = firedrake.SpatialCoordinate(mesh)

    b_0 = Constant(2440)
    δb = Constant(0)
    r_b = Constant(4791)
    expr = b_0 * exp(-inner(x, x) / r_b**2) - δb

    data = xr.open_dataset('/g/data/rd53/wy2165/heard_island/codes/island_bathymetry.nc')
    bathymetry = data['bathymetry']
    data.close()
    method = {"method": "nearest"}
    b0 = icepack.interpolate(bathymetry, S, **method)
    b = firedrake.Function(S).interpolate(expr+b0)

    loaded_values = []
    with h5py.File(os.path.join(outdir, f'hs_hybrid_{mesh_disk}.h5'), 'r') as f:
        for i in range(3001):
            loaded_values.append(f[f'{i}'][:])
             
    ############# Make the initial ice thickness.
    h = firedrake.Function(Q).interpolate(0)
    h.dat.data[:] = loaded_values[-1]

    ############# Make the SMB function.
    da_dz_abl = Constant(0.016)
    da_dz_acc = Constant(0.01)
    a_max = Constant(3.01)

    def smb(z):
        return min_value(a_max, min_value(da_dz_acc * (z - ela), da_dz_abl * (z - ela)))

    ############# Make the frontal ablation function.
    # Cell area
    cell_area = firedrake.Function(Q, name='CellArea');
    cell_area.interpolate(firedrake.CellVolume(mesh));

    # cell width
    w = firedrake.Function(Q, name='CellWidth')
    w.interpolate(firedrake.CellSize(mesh))
    
    water_level = Constant(0)
    def frontab(b,h):
        floatation_condition = firedrake.conditional(b <= - (ρ_I / ρ_W) * h, 1.0, 0.0)
        floating_mask = firedrake.Function(Q)
        floating_mask.interpolate(floatation_condition)

        return min_value(0, ((- c * h * ((ρ_I / ρ_W) * h) * w) / cell_area * (ρ_I / ρ_W)) * floating_mask) # w.e.

    s = firedrake.Function(Q).interpolate(b + h)
    a = firedrake.Function(Q).interpolate(smb(max_value(s, (1 - ρ_I / ρ_W) * h)) + frontab(b,h))

    A = icepack.rate_factor(Constant(270))
    V = firedrake.VectorFunctionSpace(mesh, cg)

    u = firedrake.Function(V)
    v = firedrake.TestFunction(V)

    P = ρ_I * g * h
    S_n = inner(grad(s), grad(s))**((n - 1) / 2)
    u_shear = -2 * A * P ** n / (n + 2) * h * S_n * grad(s)
    F = inner(u - u_shear, v) * dx

    solver_params = {'snes_type': 'ksponly', 'ksp_type': 'gmres'}
    fc_params = {'quadrature_degree': 6}
    params = {'solver_parameters': solver_params, 'form_compiler_parameters': fc_params}
    firedrake.solve(F == 0, u, **params)

    t = firedrake.Function(V).interpolate(grad(s))

    Σ = firedrake.TensorFunctionSpace(mesh, dg, symmetry=True)
    # Should this be `cg` or `dg`??
    T = firedrake.VectorFunctionSpace(mesh, cg)
    Z = V * Σ * T
    z = firedrake.Function(Z)

    z.sub(0).assign(u);

    Σ = firedrake.TensorFunctionSpace(mesh, dg, symmetry=True)
    # Should this be `cg` or `dg`??
    T = firedrake.VectorFunctionSpace(mesh, cg)
    Z = V * Σ * T
    z = firedrake.Function(Z)

    z.sub(0).assign(u);

    fns = [
        minimization.viscous_power,
        minimization.friction_power,
        minimization.momentum_balance,
    ]

        
    ############# Parameters
    τ_c = Constant(0.1)
    ε_c = Constant(Λ * (A * τ_c ** n))

    K = γ * (h * A / (n + 2))
    u_c = K * τ_c ** n + U_c

    rheology = {
        'flow_law_exponent': n,
        'flow_law_coefficient': ε_c / τ_c ** n,
        'sliding_exponent': n,
        'sliding_coefficient': u_c / τ_c ** n,
    }

    u, M, τ = firedrake.split(z)
    fields = {
        'velocity': u,
        'membrane_stress': M,
        'basal_stress': τ,
        'thickness': h,
        'surface': s,
    }

    L = sum(fn(**fields, **rheology) for fn in fns)
    F = derivative(L, z)

    h_min = Constant(1e-5)

    rfields = {
        'velocity': u,
        'membrane_stress': M,
        'basal_stress': τ,
        'thickness': firedrake.max_value(h_min, h),
        'surface': s,
    }

    L_r = sum(fn(**rfields, **rheology) for fn in fns)
    F_r = firedrake.derivative(L_r, z)
    J_r = firedrake.derivative(F_r, z)

    v_c = firedrake.replace(u_c, {h: firedrake.max_value(h, h_min)})
    linear_rheology = {
        'flow_law_exponent': 1,
        'flow_law_coefficient': ε_c / τ_c,
        'sliding_exponent': 1,
        'sliding_coefficient': v_c / τ_c,
    }

    L_1 = sum(fn(**rfields, **linear_rheology) for fn in fns)
    F_1 = firedrake.derivative(L_1, z)
    J_1 = firedrake.derivative(F_1, z)

    λ = Constant(1e-3)
    J = J_r + λ * J_1

    degree = 1
    qdegree = max(8, degree ** n)
    pparams = {'form_compiler_parameters': {'quadrature_degree': qdegree}}
    momentum_problem = firedrake.NonlinearVariationalProblem(F, z, J=J, **pparams)

    sparams = {
        'snes_type': 'newtonls',
        'snes_max_it': 200,
        'snes_linesearch_type': 'nleqerr',
        'ksp_type': 'gmres',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'umfpack',
    }
    momentum_solver = firedrake.NonlinearVariationalSolver(momentum_problem, solver_parameters=sparams)

    momentum_solver.solve()

    u, M, τ = z.subfunctions

    ϕ = firedrake.TestFunction(Q)
    G = mass_balance(thickness=h, velocity=u, accumulation=a, test_function=ϕ)
    tableau = irksome.BackwardEuler()
    t = Constant(0.0)
    dt = Constant(1.0 / 6)
    sparams = {
        'solver_parameters': {
            'ksp_type': 'gmres',
            'pc_type': 'ilu',
        },
    }
    mass_solver = irksome.TimeStepper(G, tableau, t, dt, h, **sparams)

    hs = [h.copy(deepcopy=True)]
    us = [u.copy(deepcopy=True)]

    final_time = 500.0
    num_steps = int(final_time / float(dt))
    for step in range(num_steps):
        mass_solver.advance()
        h.interpolate(max_value(0, h))
        s.interpolate(b + h)
        a.interpolate(smb(max_value(s, (1 - ρ_I / ρ_W) * h)) + frontab(b,h))

        momentum_solver.solve()

        hs.append(h.copy(deepcopy=True))
        us.append(z.subfunctions[0].copy(deepcopy=True))


    #with h5py.File(os.path.join(outdir, f'hs_Uc{U_c.dat.data[0]}_Λ{Λ}_γ{γ}_{mesh_disk}.h5'), 'w') as f:
    #with h5py.File(os.path.join(outdir, f'hs_ELA{int(ela.dat.data[0])}_{mesh_disk}.h5'), 'w') as f:
    with h5py.File(os.path.join(outdir, f'hs_ELA{int(ela.dat.data[0])}_Uc{U_c.dat.data[0]}_Λ{Λ}_γ{γ}_{mesh_disk}.h5'), 'w') as f:
    #with h5py.File(os.path.join(outdir, f'hs_ELA{int(ela.dat.data[0])}_c{c.dat.data[0]}_{mesh_disk}.h5'), 'w') as f:
        for i, thk in enumerate(hs):
            f.create_dataset(f'{i}', data=thk.dat.data[:])
            
    #with h5py.File(os.path.join(outdir, f'us_Uc{U_c.dat.data[0]}_Λ{Λ}_γ{γ}_{mesh_disk}.h5'), 'w') as f:
    #with h5py.File(os.path.join(outdir, f'us_ELA{int(ela.dat.data[0])}_{mesh_disk}.h5'), 'w') as f:
    with h5py.File(os.path.join(outdir, f'us_ELA{int(ela.dat.data[0])}_Uc{U_c.dat.data[0]}_Λ{Λ}_γ{γ}_{mesh_disk}.h5'), 'w') as f:
    #with h5py.File(os.path.join(outdir, f'us_ELA{int(ela.dat.data[0])}_c{c.dat.data[0]}_{mesh_disk}.h5'), 'w') as f:
        for i, thk in enumerate(us):
            f.create_dataset(f'{i}', data=thk.dat.data[:])
    


        
