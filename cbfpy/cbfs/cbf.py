"""
# Control Barrier Functions (CBFs)

CBFs serve as safety filters on top of a nominal controller. Given a nominal control input, the CBF will compute a
safe control input to keep the system within a safe set.

For a relative-degree-1 system, this optimizes the standard min-norm objective with the constraint
`h_dot >= -alpha(h(z))`
```
minimize ||u - u_des||_{2}^{2}               # CBF Objective (Example)
subject to Lfh(z) + Lgh(z)u >= -alpha(h(z))  # RD1 CBF Constraint
```

In the case of a relative-degree-2 system, this differs slightly to enforce the RD2 constraint
`h_2_dot >= -alpha_2(h_2(z))`
```
minimize ||u - u_des||_{2}^{2}                       # CBF Objective (Example)
subject to Lfh_2(z) + Lgh_2(z)u >= -alpha_2(h_2(z))  # RD2 CBF Constraint
```

If there are constraints on the control input, we also enforce another constraint:
```
u_min <= u <= u_max  # Control constraint
```
"""

from typing import Tuple, Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
import qpax

from cbfpy.config.cbf_config import CBFConfig
from cbfpy.utils.jax_utils import conditional_jit
from cbfpy.utils.general_utils import print_warning

# Debugging flags to disable jit in specific sections of the code.
# Note: If any higher-level jits exist, those must also be set to debug (disable jit)
DEBUG_SAFETY_FILTER = False
DEBUG_QP_DATA = False


# @jax.tree_util.register_static
@jax.tree_util.register_pytree_node_class
class CBF:
    """Control Barrier Function (CBF) class.

    The main constructor for this class is via the `from_config` method, which constructs a CBF instance
    based on the provided CBFConfig configuration object.

    You can then use the CBF's `safety_filter` method to compute the control input that satisfies the CBF

    Examples:
        ```
        # Construct a CBFConfig for your problem
        config = DroneConfig()
        # Construct a CBF instance based on the config
        cbf = CBF.from_config(config)
        # Compute the safe control input
        safe_control = cbf.safety_filter(current_state, nominal_control)
        ```
    """

    # NOTE: The __init__ method is not used to construct a CBF instance. Instead, use the `from_config` method.
    # This is because Jax prefers for the __init__ method to not contain any input validation, so we do this
    # in the CBFConfig class instead.
    def __init__(
        self,
        n: int,
        m: int,
        num_cbf: int,
        u_min: Optional[tuple],
        u_max: Optional[tuple],
        control_constrained: bool,
        relax_cbf: bool,
        cbf_relaxation_penalty: float,
        h_1: Callable[[ArrayLike], Array],
        h_2: Callable[[ArrayLike], Array],
        f: Callable[[ArrayLike], Array],
        g: Callable[[ArrayLike], Array],
        alpha: Callable[[ArrayLike], Array],
        alpha_2: Callable[[ArrayLike], Array],
        P: Callable[[ArrayLike, ArrayLike, Tuple[ArrayLike, ...]], Array],
        q: Callable[[ArrayLike, ArrayLike, Tuple[ArrayLike, ...]], Array],
        solver_tol: float,
    ):
        self.n = n
        self.m = m
        self.num_cbf = num_cbf
        self.u_min = u_min
        self.u_max = u_max
        self.control_constrained = control_constrained
        self.relax_cbf = relax_cbf
        self.cbf_relaxation_penalty = cbf_relaxation_penalty
        self.h_1 = h_1
        self.h_2 = h_2
        self.f = f
        self.g = g
        self.alpha = alpha
        self.alpha_2 = alpha_2
        self.P_config = P
        self.q_config = q
        self.solver_tol = solver_tol
        if relax_cbf:
            self.qp_solver: Callable = jax.jit(qpax.solve_qp_elastic)
        else:
            self.qp_solver: Callable = jax.jit(qpax.solve_qp)

    def tree_flatten(self):
        """Flatten the CBF object for JAX pytree registration."""
        # Return (children, aux_data) where children are the dynamic parts and aux_data are static parts
        children = ()  # No dynamic JAX arrays in this object
        aux_data = (
            self.n,
            self.m,
            self.num_cbf,
            self.u_min,
            self.u_max,
            self.control_constrained,
            self.relax_cbf,
            self.cbf_relaxation_penalty,
            self.h_1,
            self.h_2,
            self.f,
            self.g,
            self.alpha,
            self.alpha_2,
            self.P_config,
            self.q_config,
            self.solver_tol,
            self.qp_solver,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct the CBF object from flattened representation."""
        (
            n,
            m,
            num_cbf,
            u_min,
            u_max,
            control_constrained,
            relax_cbf,
            cbf_relaxation_penalty,
            h_1,
            h_2,
            f,
            g,
            alpha,
            alpha_2,
            P_config,
            q_config,
            solver_tol,
            qp_solver,
        ) = aux_data
        
        # Create a new instance without going through __init__ validation
        instance = cls.__new__(cls)
        instance.n = n
        instance.m = m
        instance.num_cbf = num_cbf
        instance.u_min = u_min
        instance.u_max = u_max
        instance.control_constrained = control_constrained
        instance.relax_cbf = relax_cbf
        instance.cbf_relaxation_penalty = cbf_relaxation_penalty
        instance.h_1 = h_1
        instance.h_2 = h_2
        instance.f = f
        instance.g = g
        instance.alpha = alpha
        instance.alpha_2 = alpha_2
        instance.P_config = P_config
        instance.q_config = q_config
        instance.solver_tol = solver_tol
        instance.qp_solver = qp_solver
        
        return instance

    @classmethod
    def from_config(cls, config: CBFConfig) -> "CBF":
        """Construct a CBF based on the provided configuration

        Args:
            config (CBFConfig): Config object for the CBF. Contains info on the system dynamics, barrier function, etc.

        Returns:
            CBF: Control Barrier Function instance
        """
        instance = cls(
            config.n,
            config.m,
            config.num_cbf,
            config.u_min,
            config.u_max,
            config.control_constrained,
            config.relax_cbf,
            config.cbf_relaxation_penalty,
            config.h_1,
            config.h_2,
            config.f,
            config.g,
            config.alpha,
            config.alpha_2,
            config.P,
            config.q,
            config.solver_tol,
        )
        instance._validate_instance(*config.init_args)
        return instance

    def _validate_instance(self, *h_args) -> None:
        """Checks that the CBF is valid; warns the user if not

        Args:
            *h_args: Optional additional arguments for the barrier function.
        """
        try:
            # TODO: Decide if this should be checked on a row-by-row basis or via the full matrix
            test_lgh = self.Lgh(jnp.ones(self.n), *h_args)
            if jnp.allclose(test_lgh, 0):
                print_warning(
                    "Lgh is zero. Consider increasing the relative degree or modifying the barrier function."
                )
        except TypeError:
            print_warning(
                "Cannot test Lgh; missing additional arguments.\n"
                + "Please provide an initial seed for these args in the config's init_args input"
            )

    @conditional_jit(not DEBUG_SAFETY_FILTER)
    def safety_filter(self, z: Array, u_des: Array, *h_args) -> Array:
        """Apply the CBF safety filter to a nominal control

        Args:
            z (Array): State, shape (n,)
            u_des (Array): Desired control input, shape (m,)
            *h_args: Optional additional arguments for the barrier function.

        Returns:
            Array: Safe control input, shape (m,)
        """
        P, q, A, b, G, h = self.qp_data(z, u_des, *h_args)
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
        if DEBUG_SAFETY_FILTER:
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

    ## QP Matrices ##

    def P_qp(  # pylint: disable=invalid-name
        self, z: Array, u_des: Array, *h_args
    ) -> Array:
        """Quadratic term in the QP objective (`minimize 0.5 * x^T P x + q^T x`)

        Args:
            z (Array): State, shape (n,)
            u_des (Array): Desired control input, shape (m,)
            *h_args: Optional additional arguments for the barrier function.

        Returns:
            Array: P matrix, shape (m, m)
        """
        # This is user-modifiable in the config, but defaults to 2 * I for the standard min-norm CBF objective
        return self.P_config(z, u_des, *h_args)

    def q_qp(self, z: Array, u_des: Array, *h_args) -> Array:
        """Linear term in the QP objective (`minimize 0.5 * x^T P x + q^T x`)

        Args:
            z (Array): State, shape (n,)
            u_des (Array): Desired control input, shape (m,)
            *h_args: Optional additional arguments for the barrier function.

        Returns:
            Array: q vector, shape (m,)
        """
        # This is user-modifiable in the config, but defaults to -2 * u_des for the standard min-norm CBF objective
        return self.q_config(z, u_des, *h_args)

    def G_qp(  # pylint: disable=invalid-name
        self, z: Array, u_des: Array, *h_args
    ) -> Array:
        """Inequality constraint matrix for the QP (`Gx <= h`)

        Note:
            The number of constraints depends on if we have control constraints or not.
                Without control constraints, `num_constraints == num_barriers`.
                With control constraints, `num_constraints == num_barriers + 2*m`

        Args:
            z (Array): State, shape (n,)
            u_des (Array): Desired control input, shape (m,)
            *h_args: Optional additional arguments for the barrier function.

        Returns:
            Array: G matrix, shape (num_constraints, m)
        """
        G = -self.Lgh(z, *h_args)
        if self.control_constrained:
            return jnp.block([[G], [jnp.eye(self.m)], [-jnp.eye(self.m)]])
        else:
            return G

    def h_qp(self, z: Array, u_des: Array, *h_args) -> Array:
        """Upper bound on constraints for the QP (`Gx <= h`)

        Note:
            The number of constraints depends on if we have control constraints or not.
                Without control constraints, `num_constraints == num_barriers`.
                With control constraints, `num_constraints == num_barriers + 2*m`

        Args:
            z (Array): State, shape (n,)
            u_des (Array): Desired control input, shape (m,)
            *h_args: Optional additional arguments for the barrier function.

        Returns:
            Array: h vector, shape (num_constraints,)
        """
        hz, lfh = self.h_and_Lfh(z, *h_args)
        h = self.alpha(hz) + lfh
        if self.control_constrained:
            return jnp.concatenate(
                [h, jnp.asarray(self.u_max), -jnp.asarray(self.u_min)]
            )
        else:
            return h

    @conditional_jit(not DEBUG_QP_DATA)
    def qp_data(
        self, z: Array, u_des: Array, *h_args
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
            u_des (Array): Desired control input, shape (m,)
            *h_args: Optional additional arguments for the barrier function.

        Returns:
            P (Array): Quadratic term in the QP objective, shape (m, m)
            q (Array): Linear term in the QP objective, shape (m,)
            A (Array): Equality constraint matrix, shape (0, m)
            b (Array): Equality constraint vector, shape (0,)
            G (Array): Inequality constraint matrix, shape (num_constraints, m)
            h (Array): Upper bound on constraints, shape (num_constraints,)
        """
        return (
            self.P_qp(z, u_des, *h_args),
            self.q_qp(z, u_des, *h_args),
            jnp.zeros((0, self.m)),  # Equality matrix (not used for CBF)
            jnp.zeros(0),  # Equality vector (not used for CBF)
            self.G_qp(z, u_des, *h_args),
            self.h_qp(z, u_des, *h_args),
        )
