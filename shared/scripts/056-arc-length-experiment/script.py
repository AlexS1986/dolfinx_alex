# arc_length_dolfinx.py
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import fem, mesh
import ufl
from dolfinx.fem import (assemble_scalar, assemble_matrix, assemble_vector,
                         apply_lifting, set_bc)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#PETSc.Sys.Print = PETSc.Log.Print  # nicer printing through PETSc

# -------------------------
# Problem setup (1D unit interval)
# -------------------------
nx = 50
domain = mesh.create_interval(comm, nx, [0.0, 1.0])
V = fem.FunctionSpace(domain, ("Lagrange", 1))

# Trial/test/unknowns
u = fem.Function(V, name="u")        # current displacement
du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Material / nonlinear coefficient k(u) = 1 + u^2
k_expr = 1.0 + u**2
# body load f(x) (we take f=1 for simplicity)
f_expr = fem.Constant(domain, PETSc.ScalarType(1.0))

# Load parameter (we will hold its value externally; not a FunctionSpace DOF)
lambda_val = 0.0

# Residual form R(u, lambda) = ∫ k(u) u' v' dx - lambda ∫ f v dx
a_internal = ufl.inner(k_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
a_external = lambda_val * ufl.inner(f_expr, v) * ufl.dx   # placeholder; we will assemble external derivative separately
R_form = a_internal - lambda_val * ufl.inner(f_expr, v) * ufl.dx

# Jacobian J = dR/du
J_form = ufl.derivative(R_form, u, du)

# Boundary conditions: u(0) = 0 (Dirichlet at left end)
def left_boundary(x):
    return np.isclose(x[0], 0.0)
bc_value = fem.Constant(domain, PETSc.ScalarType(0.0))
bc = fem.dirichletbc(bc_value, fem.locate_dofs_geometrical(V, left_boundary), V)
bcs = [bc]

# Helper functions to assemble residual and tangent and external load vector
def assemble_residual_and_tangent(u_func, lambda_scalar):
    """
    Assemble residual vector R and Jacobian matrix J (PETSc) for given u and lambda.
    Also returns the "external load" vector F_ext = -dR/dlambda (PETSc Vec).
    """
    # Update forms that depend on lambda: build R_form and J_form freshly using current lambda
    # (we rebuild symbolic forms here to ensure the lambda value is used in the residual)
    current_R_form = ufl.inner((1.0 + u_func**2) * ufl.grad(u_func), ufl.grad(v)) * ufl.dx - lambda_scalar * ufl.inner(f_expr, v) * ufl.dx
    current_J_form = ufl.derivative(current_R_form, u_func, du)

    # Assemble tangent matrix
    A = assemble_matrix(current_J_form, bcs=bcs)
    A.assemble()

    # Assemble residual vector
    b = assemble_vector(current_R_form)
    # apply lifting and BC corrections (standard dolfinx pattern)
    apply_lifting(b, [current_J_form], [bcs])
    for bc in bcs:
        bc.zero(b)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Assemble external load derivative dR/dlambda = -∫ f v dx
    # Compute F_ext = - dR/dlambda: assemble vector for ∫ f v dx, then multiply by -1
    ext_form = ufl.inner(f_expr, v) * ufl.dx
    F_ext = assemble_vector(ext_form)
    # No apply_lifting for this simple external vector; apply BC zeroing
    for bc in bcs:
        bc.zero(F_ext)
    F_ext.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    # dR/dlambda = -F_ext
    F_ext.scale(-1.0)

    return A, b, F_ext

# --- Arc-length parameters ---
alpha = 1.0  # scaling between displacement and lambda in constraint
delta_s = 0.1  # initial arc length
max_steps = 80
tol_res = 1e-8
tol_arc = 1e-8
max_corrector_iters = 20

# storage of previous converged solution (start from u=0, lambda=0)
u_prev = fem.Function(V)
u_prev.x.array[:] = 0.0
lambda_prev = 0.0

# current solution (start same as prev)
u.x.array[:] = u_prev.x.array[:]
lambda_current = lambda_prev

# previous increments (for predictor/extrapolation)
delta_u_prev = np.zeros_like(u.x.array)
delta_lambda_prev = 0.0

# For printing
def print_step(step, lam, resnorm):
    if rank == 0:
        print(f"[step {step}] lambda = {lam:.6g}  ||R||_inf = {resnorm:.3e}")

# ---------- continuation loop ----------
for step in range(1, max_steps + 1):
    # Predictor: simple secant extrapolation if available, else take lambda step
    if step == 1:
        # first predictor: small lambda increment direction + internal extrapolation
        delta_lambda_pred = delta_s / alpha
        delta_u_pred = delta_u_prev.copy()  # zeros
    else:
        # secant-like predictor: use previous increment
        delta_lambda_pred = delta_lambda_prev
        delta_u_pred = delta_u_prev

    # apply predictor
    u.x.array[:] = u_prev.x.array + delta_u_pred
    lambda_predict = lambda_prev + delta_lambda_pred

    # Initialize total increments for current step (from last converged state to current iterate)
    Delta_u = u.x.array.copy() - u_prev.x.array  # numpy array
    Delta_lambda = lambda_predict - lambda_prev

    # Corrector: Newton iterations on augmented system
    converged = False
    for corr_iter in range(1, max_corrector_iters + 1):
        # Assemble tangent, residual, and F_ext at current u and lambda_predict + corrections
        A_tan, R_vec, F_ext_vec = assemble_residual_and_tangent(u, lambda_predict)

        # Compute residual norm
        R_arr = R_vec.getArray(readonly=True)
        res_norm = np.max(np.abs(R_arr))

        # Build augmented PETSc matrix of size (ndof+1) x (ndof+1)
        n = A_tan.getSize()[0]
        # Create new empty square matrix (AIJ) for augmented system
        aug_mat = PETSc.Mat().create(comm=comm)
        aug_mat.setSizes(((n+1, None), (n+1, None)))
        aug_mat.setType("aij")
        # Preallocate: take row nnz pattern from A_tan diagonals (simple approach)
        # For small problems this is fine; otherwise use Preallocation properly
        aug_mat.setUp()

        # Insert A_tan into top-left block of aug_mat
        # Extract A_tan as PETSc Mat and copy values (MatCopy would be easier if same layout)
        aug_mat.zeroEntries()
        aug_mat.assemble()
        # Copy entries from A_tan into aug_mat
        # We use Mat.getValuesCSR and setValuesCSR to efficiently copy
        Ai, Aj, Av = A_tan.getValuesCSR()
        aug_mat.setValuesCSR(Ai, Aj, Av)  # places into rows 0..n-1, cols 0..n-1
        # Now we need to insert the right-hand column (top n rows): -F_ext_vec
        # But note F_ext_vec is size n PETSc Vec. We'll set column n with -F_ext entries.
        # We'll create arrays of row indices and values for that column.
        farr = F_ext_vec.getArray(readonly=True)
        # Put -F_ext into column n at rows 0..n-1
        rows = np.arange(n, dtype=np.int32)
        cols = np.full(n, n, dtype=np.int32)
        aug_mat.setValues(rows.tolist(), cols.tolist(), (-farr).tolist(), addv=PETSc.InsertMode.ADD)

        # Insert bottom row: (Delta_u^T) across columns 0..n-1, and alpha^2 at (n,n)
        # Delta_u is a numpy array in DOF ordering; ensure ghost updates already done (our u is local)
        Delta_u_vec = Delta_u.copy()
        # bottom row entries
        aug_mat.setValues([n], list(range(n)), Delta_u_vec.tolist(), addv=PETSc.InsertMode.ADD)
        # bottom-right scalar
        aug_mat.setValue(n, n, alpha**2, addv=PETSc.InsertMode.ADD)

        aug_mat.assemble()

        # Build augmented RHS vector of size n+1:
        aug_rhs = PETSc.Vec().createMPI(size=n+1, comm=comm)
        # top n entries are -R_vec
        Rarr = R_vec.getArray(readonly=True)
        aug_rhs.setValues(range(n), (-Rarr).tolist(), addv=PETSc.InsertMode.INSERT)
        # bottom entry is the constraint residual:
        # c = Delta_u^T * Delta_u + alpha^2 * Delta_lambda - delta_s^2
        c_val = float(Delta_u.dot(Delta_u) + alpha**2 * Delta_lambda - (delta_s**2))
        aug_rhs.setValue(n, c_val, addv=PETSc.InsertMode.INSERT)
        aug_rhs.assemble()

        # Solve augmented linear system with PETSc KSP
        ksp = PETSc.KSP().create(comm=comm)
        ksp.setOperators(aug_mat)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")   # for small problems use LU direct solver
        ksp.setFromOptions()
        x_aug = aug_rhs.duplicate()
        ksp.solve(aug_rhs, x_aug)
        # x_aug now contains [delta_u(0..n-1); delta_lambda]

        # Extract corrections
        delta_u_correction = x_aug.getArray(readonly=True)[:n].copy()
        delta_lambda_correction = float(x_aug.getArray(readonly=True)[n])

        # Update solution and total increments
        # update u vector (PETSc Vec underlying Function)
        u.x.array[:] = u.x.array + delta_u_correction
        lambda_predict = lambda_predict + delta_lambda_correction
        Delta_u = u.x.array - u_prev.x.array
        Delta_lambda = lambda_predict - lambda_prev

        # Convergence check: residual norm and arc residual
        # Recompute residual norm quickly (we already have Rarr for previous u, but after update we should recompute)
        A_tan_new, R_vec_new, F_ext_new = assemble_residual_and_tangent(u, lambda_predict)
        Rarr_new = R_vec_new.getArray(readonly=True)
        res_norm_new = np.max(np.abs(Rarr_new))
        arc_res = abs(Delta_u.dot(Delta_u) + alpha**2 * Delta_lambda - (delta_s**2))

        if rank == 0:
            print(f"  corr {corr_iter}: ||R||={res_norm_new:.3e}, arc_res={arc_res:.3e}, lambda={lambda_predict:.6g}")

        if res_norm_new < tol_res and arc_res < tol_arc:
            converged = True
            # Set current lambda
            lambda_current = lambda_predict
            # Update u (already updated), break corrector loop
            break

        # otherwise continue corrector iterations

    # End corrector iterations
    if not converged:
        # simple strategy: reduce step size and retry (for minimal example)
        if rank == 0:
            print(f"Step {step}: corrector did not converge. Reducing delta_s and retrying.")
        delta_s *= 0.5
        # restore previous converged solution and try again from start of this step
        u.x.array[:] = u_prev.x.array[:]
        lambda_predict = lambda_prev
        Delta_u = np.zeros_like(Delta_u)
        Delta_lambda = 0.0
        # Also reset previous increments to be safe
        delta_u_pred = np.zeros_like(delta_u_pred)
        delta_lambda_pred = 0.0
        # Try this step again (decrement step index)
        # For simplicity in this minimal example, break the continuation on failure
        break

    # Accept step: store previous solution and increments
    u_prev.x.array[:] = u.x.array[:]
    lambda_prev = lambda_current
    delta_u_prev = (u_prev.x.array - u_prev.x.array) if False else (u.x.array - (u_prev.x.array - Delta_u))  # keep for predictor; simple approach
    # Better: use the actual step increment just taken:
    delta_u_prev = u.x.array - (u_prev.x.array - 0.0)  # here u_prev already updated, but we want the increment used
    # Simpler: store the increment from last two points (for next predictor)
    # For this minimal example, use the last increment
    delta_u_prev = Delta_u.copy()
    delta_lambda_prev = Delta_lambda

    # Print status
    print_step(step, lambda_current, res_norm_new)

    # Optionally adapt delta_s: increase if converged fast
    if corr_iter <= 3:
        delta_s *= 1.2
    elif corr_iter >= 8:
        delta_s *= 0.5

# End continuation
if rank == 0:
    print("Continuation finished.")
    print("Final lambda:", lambda_current)
