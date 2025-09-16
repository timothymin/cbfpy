"""
# Control Lyapunov Function / Control Barrier Functions (CLF-CBFs)

Whereas a CBF acts as a safety filter on top of a nominal controller, a CLF-CBF acts as a safe controller itself, 
based on a control objective defined by the CLF and a safety constraint defined by the CBF. Note that the CLF
objective should be quadratic and positive-definite to fit in this QP framework.

The CLF-CBF optimizes the following:
```
minimize   ||u||_{2}^{2}                     # CLF Objective (Example)
subject to LfV(z) + LgV(z)u <= -gamma(V(z))  # CLF Constraint
           Lfh(z) + Lgh(z)u >= -alpha(h(z))  # CBF Constraint
```

As with the CBF, if this is a relative-degree-2 system, we update the constraints:
```
minimize   ||u||_{2}^{2}                             # CLF Objective (Example)
subject to LfV_2(z) + LgV_2(z)u <= -gamma_2(V_2(z))  # RD2 CLF Constraint
           Lfh_2(z) + Lgh_2(z)u >= -alpha_2(h_2(z))  # RD2 CBF Constraint
```

If there are constraints on the control input, we also enforce another constraint:
```
u_min <= u <= u_max  # Control constraint
```

However, in general the CLF constraint and the CBF constraint cannot be strictly enforced together. We then
need to introduce a slack variable to relax the CLF constraint, ensuring that the CBF safety condition takes
priority over the CLF objective.

The optimization problem then becomes:
```
minimize   ||u||_{2}^{2} + p * delta^2               # CLF Objective (Example)
subject to LfV(z) + LgV(z)u <= -gamma(V(z)) + delta  # CLF Constraint
           Lfh(z) + Lgh(z)u >= -alpha(h(z))          # CBF Constraint
```
where `p` is a large constant and `delta` is the slack variable.
"""

from typing import Tuple, Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
import qpax

from cbfpy.config.clf_cbf_config import CLFCBFConfig
from cbfpy.utils.jax_utils import conditional_jit
from cbfpy.utils.general_utils import print_warning

# Debugging flags to disable jit in specific sections of the code.
# Note: If any higher-level jits exist, those must also be set to debug (disable jit)
DEBUG_CONTROLLER = False
DEBUG_QP_DATA = False


# @jax.tree_util.register_static
@jax.tree_util.register_pytree_node_class
class CLFCBF:
    """Control Lyapunov Function / Control Barrier Function (CLF-CBF) class.

    The main constructor for this class is via the `from_config` method, which constructs a CLF-CBF instance
    based on the provided CLFCBFConfig configuration object.

    You can then use the CLF-CBF's `controller` method to compute the optimal control input

    Examples:
        ```
        # Construct a CLFCBFConfig for your problem
        config = DroneConfig()
        # Construct a CBF instance based on the config
        clf_cbf = CLFCBF.from_config(config)
        # Compute the safe control input
        safe_control = clf_cbf.controller(current_state, desired_state)
        ```
    """

    # NOTE: The __init__ method is not used to construct a CLFCBF instance. Instead, use the `from_config` method.
    # This is because Jax prefers for the __init__ method to not contain any input validation, so we do this
    # in the CLFCBFConfig class instead.
    def __init__(
        self,
        n: int,
        m: int,
        num_cbf: int,
        num_clf: int,
        u_min: Optional[tuple],
        u_max: Optional[tuple],
        control_constrained: bool,
        relax_cbf: bool,
        cbf_relaxation_penalty: float,
        clf_relaxation_penalty: float,
        h_1: Callable[[ArrayLike], Array],
        h_2: Callable[[ArrayLike], Array],
        f: Callable[[ArrayLike], Array],
        g: Callable[[ArrayLike], Array],
        alpha: Callable[[ArrayLike], Array],
        alpha_2: Callable[[ArrayLike], Array],
        V_1: Callable[[ArrayLike], Array],
        V_2: Callable[[ArrayLike], Array],
        gamma: Callable[[ArrayLike], Array],
        gamma_2: Callable[[ArrayLike], Array],
        H: Callable[[ArrayLike], Array],
        F: Callable[[ArrayLike], Array],
        solver_tol: float,
    ):
        self.n = n
        self.m = m
        self.num_cbf = num_cbf
        self.num_clf = num_clf
        self.u_min = u_min
        self.u_max = u_max
        self.control_constrained = control_constrained
        self.relax_cbf = relax_cbf
        self.cbf_relaxation_penalty = cbf_relaxation_penalty
        self.clf_relaxation_penalty = clf_relaxation_penalty
        self.h_1 = h_1
        self.h_2 = h_2
        self.f = f
        self.g = g
        self.alpha = alpha
        self.alpha_2 = alpha_2
        self.V_1 = V_1
        self.V_2 = V_2
        self.gamma = gamma
        self.gamma_2 = gamma_2
        self.H = H
        self.F = F
        self.solver_tol = solver_tol
        if relax_cbf:
            self.qp_solver: Callable = jax.jit(qpax.solve_qp_elastic)
        else:
            self.qp_solver: Callable = jax.jit(qpax.solve_qp)

    def tree_flatten(self):
        """Flatten the CLFCBF object for JAX pytree registration."""
        # Return (children, aux_data) where children are the dynamic parts and aux_data are static parts
        children = ()  # No dynamic JAX arrays in this object
        aux_data = (
            self.n,
            self.m,
            self.num_cbf,
            self.num_clf,
            self.u_min,
            self.u_max,
            self.control_constrained,
            self.relax_cbf,
            self.cbf_relaxation_penalty,
            self.clf_relaxation_penalty,
            self.h_1,
            self.h_2,
            self.f,
            self.g,
            self.alpha,
            self.alpha_2,
            self.V_1,
            self.V_2,
            self.gamma,
            self.gamma_2,
            self.H,
            self.F,
            self.solver_tol,
            self.qp_solver,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct the CLFCBF object from flattened representation."""
        (
            n,
            m,
            num_cbf,
            num_clf,
            u_min,
            u_max,
            control_constrained,
            relax_cbf,
            cbf_relaxation_penalty,
            clf_relaxation_penalty,
            h_1,
            h_2,
            f,
            g,
            alpha,
            alpha_2,
            V_1,
            V_2,
            gamma,
            gamma_2,
            H,
            F,
            solver_tol,
            qp_solver,
        ) = aux_data
        
        # Create a new instance without going through __init__ validation
        instance = cls.__new__(cls)
        instance.n = n
        instance.m = m
        instance.num_cbf = num_cbf
        instance.num_clf = num_clf
        instance.u_min = u_min
        instance.u_max = u_max
        instance.control_constrained = control_constrained
        instance.relax_cbf = relax_cbf
        instance.cbf_relaxation_penalty = cbf_relaxation_penalty
        instance.clf_relaxation_penalty = clf_relaxation_penalty
        instance.h_1 = h_1
        instance.h_2 = h_2
        instance.f = f
        instance.g = g
        instance.alpha = alpha
        instance.alpha_2 = alpha_2
        instance.V_1 = V_1
        instance.V_2 = V_2
        instance.gamma = gamma
        instance.gamma_2 = gamma_2
        instance.H = H
        instance.F = F
        instance.solver_tol = solver_tol
        instance.qp_solver = qp_solver
        
        return instance

    @classmethod
    def from_config(cls, config: CLFCBFConfig) -> "CLFCBF":
        """Construct a CLF-CBF based on the provided configuration

        Args:
            config (CLFCBFConfig): Config object for the CLF-CBF. Contains info on the system dynamics, barrier
                function, Lyapunov function, etc.

        Returns:
            CLFCBF: Control Lyapunov Function / Control Barrier Function instance
        """
        instance = cls(
            config.n,
            config.m,
            config.num_cbf,
            config.num_clf,
            config.u_min,
            config.u_max,
            config.control_constrained,
            config.relax_cbf,
            config.cbf_relaxation_penalty,
            config.clf_relaxation_penalty,
            config.h_1,
            config.h_2,
            config.f,
            config.g,
            config.alpha,
            config.alpha_2,
            config.V_1,
            config.V_2,
            config.gamma,
            config.gamma_2,
            config.H,
            config.F,
            config.solver_tol,
        )
        instance._validate_instance(*config.init_args)
        return instance

    def _validate_instance(self, *h_args) -> None:
        """Checks that the CLF-CBF is valid; warns the user if not

        Args:
            *h_args: Optional additional arguments for the barrier function.
        """
        test_z = jnp.ones(self.n)
        try:
            test_lgh = self.Lgh(test_z, *h_args)
            if jnp.allclose(test_lgh, 0):
                print_warning(
                    "Lgh is zero. Consider increasing the relative degree or modifying the barrier function."
                )
        except TypeError:
            print_warning(
                "Cannot test Lgh; missing additional arguments.\n"
                + "Please provide an initial seed for these args in the config's init_args input"
            )
        test_lgv = self.LgV(test_z, test_z)
        if jnp.allclose(test_lgv, 0):
            print_warning(
                "LgV is zero. Consider increasing the relative degree or modifying the Lyapunov function."
            )

    @conditional_jit(not DEBUG_CONTROLLER)
    def controller(self, z: Array, z_des: Array, *h_args) -> Array:
        """Compute the CLF-CBF optimal control input, optimizing for the CLF objective while
        satisfying the CBF safety constraint.

        Args:
            z (Array): State, shape (n,)
            z_des (Array): Desired state, shape (n,)
            *h_args: Optional additional arguments for the barrier function.

        Returns:
            Array: Safe control input, shape (m,)
        """
        P, q, A, b, G, h = self.qp_data(z, z_des, *h_args)
        if self.relax_cbf:
            x_qp, t_qp, s1_qp, s2_qp, z1_qp, z2_qp, converged, iters = self.qp_solver(
                P,
                q,
                G,
                h,
                self.cbf_relaxation_penalty,
                solver_tol=self.solver_tol,
            )
        else:
            x_qp, s_qp, z_qp, y_qp, converged, iters = self.qp_solver(
                P,
                q,
                A,
                b,
                G,
                h,
                solver_tol=self.solver_tol,
            )
        if DEBUG_CONTROLLER:
            print(
                f"{'Converged' if converged else 'Did not converge'}. Iterations: {iters}"
            )
        return x_qp[: self.m]

    def h(self, z: ArrayLike, *h_args) -> Array:
        """Barrier function(s)

        Args:
            z (ArrayLike): State, shape (n,)
            *h_args: Optional additional arguments for the barrier function.

        Returns:
            Array: Barrier function evaluation, shape (num_barr,)
        """

        # Take any relative-degree-2 barrier functions and convert them to relative-degree-1
        def _h_2(state):
            return self.h_2(state, *h_args)

        h_2, dh_2_dt = jax.jvp(_h_2, (z,), (self.f(z),))
        h_2_as_rd1 = dh_2_dt + self.alpha_2(h_2)

        # Merge the relative-degree-1 and relative-degree-2 barrier functions
        return jnp.concatenate([self.h_1(z, *h_args), h_2_as_rd1])

    def h_and_Lfh(  # pylint: disable=invalid-name
        self, z: ArrayLike, *h_args
    ) -> Tuple[Array, Array]:
        """Lie derivative of the barrier function(s) wrt the autonomous dynamics `f(z)`

        The evaluation of the barrier function is also returned "for free", a byproduct of the jacobian-vector-product

        Args:
            z (ArrayLike): State, shape (n,)
            *h_args: Optional additional arguments for the barrier function.

        Returns:
            h (Array): Barrier function evaluation, shape (num_barr,)
            Lfh (Array): Lie derivative of `h` w.r.t. `f`, shape (num_barr,)
        """
        # Note: the below code is just a more efficient way of stating `Lfh = jax.jacobian(self.h)(z) @ self.f(z)`
        # with the bonus benefit of also evaluating the barrier function

        def _h(state):
            return self.h(state, *h_args)

        return jax.jvp(_h, (z,), (self.f(z),))

    def Lgh(self, z: ArrayLike, *h_args) -> Array:  # pylint: disable=invalid-name
        """Lie derivative of the barrier function(s) wrt the control dynamics `g(z)u`

        Args:
            z (ArrayLike): State, shape (n,)
            *h_args: Optional additional arguments for the barrier function.

        Returns:
            Array: Lgh, shape (num_barr, m)
        """
        # Note: the below code is just a more efficient way of stating `Lgh = jax.jacobian(self.h)(z) @ self.g(z)`

        def _h(state):
            return self.h(state, *h_args)

        def _jvp(g_column):
            return jax.jvp(_h, (z,), (g_column,))[1]

        return jax.vmap(_jvp, in_axes=1, out_axes=1)(self.g(z))

    ## CLF functions ##

    def V(self, z: ArrayLike, z_des: ArrayLike) -> Array:
        """Control Lyapunov Function(s)

        Args:
            z (ArrayLike): State, shape (n,)
            z_des (ArrayLike): Desired state, shape (n,)

        Returns:
            Array: CLF evaluation, shape (num_clf,)
        """
        def _V_2(state):
            return self.V_2(state, z_des)

        # Take any relative-degree-2 CLFs and convert them to relative-degree-1
        # NOTE: If adding args to the CLF, create a wrapper func like with the barrier function
        V_2, dV_2_dt = jax.jvp(_V_2, (z,), (self.f(z),))
        V2_rd1 = dV_2_dt + self.gamma_2(V_2)

        # Merge the relative-degree-1 and relative-degree-2 CLFs
        return jnp.concatenate([self.V_1(z, z_des), V2_rd1])

    def V_and_LfV(self, z: ArrayLike, z_des: ArrayLike) -> Tuple[Array, Array]:
        """Lie derivative of the CLF wrt the autonomous dynamics `f(z)`

        The evaluation of the CLF is also returned "for free", a byproduct of the jacobian-vector-product

        Args:
            z (ArrayLike): State, shape (n,)
            z_des (ArrayLike): Desired state, shape (n,)

        Returns:
            V (Array): CLF evaluation, shape (1,)
            LfV (Array): Lie derivative of `V` w.r.t. `f`, shape (1,)
        """

        def _V(state):
            return self.V(state, z_des)

        return jax.jvp(_V, (z,), (self.f(z),))

    def LgV(self, z: ArrayLike, z_des: ArrayLike) -> Array:
        """Lie derivative of the CLF wrt the control dynamics `g(z)u`

        Args:
            z (ArrayLike): State, shape (n,)
            z_des (ArrayLike): Desired state, shape (n,)

        Returns:
            Array: LgV, shape (m,)
        """

        def _V(state):
            return self.V(state, z_des)

        def _jvp(g_column):
            return jax.jvp(_V, (z,), (g_column,))[1]

        return jax.vmap(_jvp, in_axes=1, out_axes=1)(self.g(z))

    ## QP Matrices ##

    def P_qp(  # pylint: disable=invalid-name
        self, z: Array, z_des: Array, *h_args
    ) -> Array:
        """Quadratic term in the QP objective (`minimize 0.5 * x^T P x + q^T x`)

        Args:
            z (Array): State, shape (n,)
            z_des (Array): Desired state, shape (n,)
            *h_args: Optional additional arguments for the barrier function.

        Returns:
            Array: P matrix, shape (m, m)
        """
        return jnp.block(
            [
                [self.H(z), jnp.zeros((self.m, 1))],
                [jnp.zeros((1, self.m)), jnp.atleast_1d(self.clf_relaxation_penalty)],
            ]
        )

    def q_qp(self, z: Array, z_des: Array, *h_args) -> Array:
        """Linear term in the QP objective (`minimize 0.5 * x^T P x + q^T x`)

        Args:
            z (Array): State, shape (n,)
            z_des (Array): Desired state, shape (n,)
            *h_args: Optional additional arguments for the barrier function.

        Returns:
            Array: Q vector, shape (m,)
        """
        return jnp.concatenate([self.F(z), jnp.array([0.0])])

    def G_qp(  # pylint: disable=invalid-name
        self, z: Array, z_des: Array, *h_args
    ) -> Array:
        """Inequality constraint matrix for the QP (`Gx <= h`)

        Note:
            The number of constraints depends on if we have control constraints or not.
                Without control constraints, `num_constraints == num_barriers`.
                With control constraints, `num_constraints == num_barriers + 2*m`

        Args:
            z (Array): State, shape (n,)
            z_des (Array): Desired state, shape (n,)
            *h_args: Optional additional arguments for the barrier function.

        Returns:
            Array: G matrix, shape (num_constraints, m)
        """
        G = jnp.block(
            [
                [self.LgV(z, z_des), -1.0 * jnp.ones((self.num_clf, 1))],
                [-self.Lgh(z, *h_args), jnp.zeros((self.num_cbf, 1))],
            ]
        )
        if self.control_constrained:
            return jnp.block(
                [
                    [G],
                    [jnp.eye(self.m), jnp.zeros((self.m, 1))],
                    [-jnp.eye(self.m), jnp.zeros((self.m, 1))],
                ]
            )
        else:
            return G

    def h_qp(self, z: Array, z_des: Array, *h_args) -> Array:
        """Upper bound on constraints for the QP (`Gx <= h`)

        Note:
            The number of constraints depends on if we have control constraints or not.
                Without control constraints, `num_constraints == num_barriers`.
                With control constraints, `num_constraints == num_barriers + 2*m`

        Args:
            z (Array): State, shape (n,)
            z_des (Array): Desired state, shape (n,)
            *h_args: Optional additional arguments for the barrier function.

        Returns:
            Array: h vector, shape (num_constraints,)
        """
        hz, lfh = self.h_and_Lfh(z, *h_args)
        vz, lfv = self.V_and_LfV(z, z_des)
        h = jnp.concatenate(
            [
                -lfv - self.gamma(vz),
                self.alpha(hz) + lfh,
            ]
        )
        if self.control_constrained:
            return jnp.concatenate(
                [h, jnp.asarray(self.u_max), -jnp.asarray(self.u_min)]
            )
        else:
            return h

    @conditional_jit(not DEBUG_QP_DATA)
    def qp_data(
        self, z: Array, z_des: Array, *h_args
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """Constructs the QP matrices based on the current state and desired control

        i.e. the matrices/vectors (P, q, A, b, G, h) for the optimization problem:

        ```
        minimize 0.5 * x^T P x + q^T x
        subject to  A x == b
                    G x <= h
        ```

        Note:
            - CBFs do not rely on equality constraints, so `A` and `b` are empty.
            - The number of constraints depends on if we have control constraints or not.
                Without control constraints, `num_constraints == num_barriers`.
                With control constraints, `num_constraints == num_barriers + 2*m`

        Args:
            z (Array): State, shape (n,)
            z_des (Array): Desired state, shape (n,)
            *h_args: Optional additional arguments for the barrier function.

        Returns:
            P (Array): Quadratic term in the QP objective, shape (m + 1, m + 1)
            q (Array): Linear term in the QP objective, shape (m + 1,)
            A (Array): Equality constraint matrix, shape (0, m + 1)
            b (Array): Equality constraint vector, shape (0,)
            G (Array): Inequality constraint matrix, shape (num_constraints, m + 1)
            h (Array): Upper bound on constraints, shape (num_constraints,)
        """
        return (
            self.P_qp(z, z_des, *h_args),
            self.q_qp(z, z_des, *h_args),
            jnp.zeros((0, self.m + 1)),  # Equality matrix (not used for CLF-CBF)
            jnp.zeros(0),  # Equality vector (not used for CLF-CBF)
            self.G_qp(z, z_des, *h_args),
            self.h_qp(z, z_des, *h_args),
        )
