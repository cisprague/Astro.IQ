---
title: Getting Started
permalink: /getting_started/
---

## Defining a Dynamical Model
In **Astro.IQ**, all models are initialised based off of the `Dynamical_Model` class:

```python
class Dynamical_Model(object):
  def __init__(self, si, st, slb, sub, clb, cub, tlb, tub):
  # For Direct Methods
  def EOM_State(self, state, control):
  def EOM_State_Jac(self, state, control):
  # For Indirect Methods
  def EOM_Fullstate(self, fullstate, control):
  def EOM_Fullstate_Jac(self, fullstate, control):
  def Hamiltonian(self, fullstate):
  def Pontryagin(self, fullstate):
```

### Parameters
The system's state is described by the iterables `si` and `st`, describing the system's initial and target *state* respectively, which are constrained by lower and upper *bounds*, `slb` and `sub`, of equal dimension, i.e. `shape(si) == shape(slb)` and v.v.. Similarly, while the *control* is not yet defined, it is bounded by the iterables `clb` and `cub` of equal dimension to the problem's control space dimensionality. One should note the the dimensionality of the *state* and *control* space is uniquely determined by the problem itself. Lastly, because each model environment can be vastly different, it is required to specify the lower and upper bounds of the trajectory's duration with the *floats* `tlb` and `tub`.

#### Example
To further elaborate on this methodology, consider a two-dimensional planetary lander, modelled as a point mass capable of thrusting in any direction, given by the set first order ordinary differential equations

$$
  \dot{\pmb{s}} =
  \begin{bmatrix}
  \dot{x} \\
  \dot{y} \\
  \dot{v}_x \\
  \dot{v}_y \\
  \dot{m}
  \end{bmatrix}
  =
  \begin{bmatrix}
  v_x \\
  v_y \\
  \frac{T u}{m} \hat{u}_x \\
  \frac{T u}{m} \hat{u}_y - g\\
  -\frac{T}{I_{sp}g_0}
  \end{bmatrix}
$$

, where the set of parameters $$\{T, I_{sp}, g_0, g\}$$, maximum thrust, specific impulse, Earth's sea-level gravity, and the environmental gravity, are inherent to this specific model. This *dynamical model* in particular has a *state space* $$\mathcal{S}$$ dimensionality of 5 given by the vector

$$
\pmb{s}^\intercal = [x, y, v_x, v_y, m] \in \mathcal{S}
$$

and a *control space* $$\mathcal{U}$$ dimensionality of 3 given by the vector

$$
\pmb{c}^\intercal = [u, \hat{u}_x, \hat{u}_y] \in \mathcal{U}
$$

One can define such a model as follows:

```python
class Lander(Dynamical_Model):
    def __init__(
        self,                           # Allowing the
        si  = [10, 1000, 20, -5, 9500], # initial state,
        st  = [0, 0, 0, 0, 8000],       # target state,
        Isp = 311,                      # and parametres
        g   = 1.6229,                   # to be reinstantiated
        T   = 44000                     # as the user wishes
    ):

        # Problem parametres
        self.Isp  = float(Isp)   # Specific impulse [s]
        self.g    = float(g)     # Environment's gravity [m/s^2]
        self.T    = float(T)     # Maximum thrust [N]
        self.g0   = float(9.802) # Earth's sea-level gravity [m/s^2]

        # Instantiate as a Dynamical_Model
        Dynamical_Model.__init__(
            self,
            si,                            # Initial state
            st,                            # Target state
            [-1000, 0, -500, -500, 0],     # State lower bounds
            [1000, 2000, 500, 500, 10000], # State upper bounds
            [0, -1, -1],                   # Control lower bounds
            [1, 1, 1],                     # Control upper bounds
            1,                             # Time lower bounds
            200                            # Time upper bounds
        )
```

### Equations of Motion
The model must be instantiated having at least the instance method `EOM_State`, describing it's equations of motion. Referring back to the above definition of $$\dot{\pmb{s}}$$, one can do so as follows

```python
def EOM_State(self, state, control):
    x, y, vx, vy, m = state
    u, ux, uy       = control
    x0 = self.T*u/m
    return array([
        vx,
        vy,
        ux*x0,
        uy*x0 - self.g,
        -self.T*u/(self.Isp*self.g0)
    ], float)
```

In order to improve the speed and accuracy of gradient based optimisation methods and the a numerical integration, one can optionally also define the square *Jacobian* matrix $$ \frac{\partial \dot{\pmb{s}}}{\partial \pmb{s}} $$ for the equations of motion, which for the planetary lander is

$$
\frac{\partial \dot{\pmb{s}}}{\partial \pmb{s}}
=
\begin{bmatrix}
0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & - \frac{T \hat{u}_x}{m^{2}} u \\
0 & 0 & 0 & 0 & - \frac{T \hat{u}_y}{m^{2}} u \\
0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

which is implemented with the instance method

```python
def EOM_State_Jac(self, state, control):
    x, y, vx, vy, m = state
    u, st, ct       = control
    x0              = self.T*u/m**2
    return array([
        [0, 0, 1, 0,        0],
        [0, 0, 0, 1,        0],
        [0, 0, 0, 0, -ux*x0/m],
        [0, 0, 0, 0, -uy*x0/m],
        [0, 0, 0, 0,        0]
    ], float)
```

Just defining a model with the instance method `EOM_State` alone is enough to perform trajectory optimisation using *direct methods*.

### Full State Equations of Motion
If one wishes to use *indirect methods* to solve a trajectory optimisation problem, it is necessary to define the instance methods `EOM_Fullstate`, `Hamiltonian`, and `Pontryagin`, which require analytical derivation through optimal control theory. Starting out, one first defines the vector of *costate* variables $$\pmb{\lambda}$$ or `l` corresponding to the system's state such that `shape(l) == shape(s)`. In the particular case of the lander described above, this would be

$$
\pmb{\lambda}^\intercal = [\lambda_x, \lambda_y, \lambda_{v_x}, \lambda_{v_y}, \lambda_m]
$$

, from which the *Hamiltonian* is derived through

$$ \mathcal{H} = \pmb{\lambda}^\intercal \dot{\pmb{s}} + \mathcal{L} $$

, where the system's *Lagrangian* or *cost functional*, in this model, is defined as

$$
\mathcal{L} = T u
$$

, which for this model basically says that the *optimal trajectory* should minimise the fuel expenditure. The *Hamiltonian* for this model becomes

$$
\begin{multline}
\mathcal{H} =
\lambda_x v_x +
\lambda_y v_y + \\
\lambda_{v_x} \left( \frac{T u}{m} \hat{u}_x \right) +
\lambda_{v_x} \left( \frac{T u}{m} \hat{u}_y - g \right) + \\
\lambda_m \left( -\frac{T}{I_{sp}g_0} \right) +
T u
\end{multline}
$$

which is then defined with the instance method

```python
def Hamiltonian(self, fullstate, control):
    x, y, vx, vy, m, lx, ly, lvx, lvy, lm = fullstate
    u, ux, uy = control
    # Get the model parametres
    T, Isp, g0, g = self.T, self.Isp, self.g0, self.g
    # Common sub expression elimination
    x0 = T*u/m
    x1 = 1/(Isp*g0)
    # Dot the costates with the states
    H  = lx*vx + ly*vy + lvx*ux*x0 + lvy*(uy*x0 - g) - T*lvm*u*x1
    # Add the Lagrangian or cost functional
    H += T*u
    return H
```

After the *Hamiltonian* is derived, one then derives the *costate equations motion*, which through optimal control theory is found through

$$
\dot{\pmb{\lambda}} = - \frac{\partial H}{\partial \pmb{s}}
$$

, which for this model becomes

$$
\dot{\pmb{\lambda}} =
\begin{bmatrix}
0\\
0\\
- \lambda_x\\
- \lambda_y\\
\frac{T u}{m^2} \left(
  \hat{u}_x \lambda_{v_x} + \hat{u}_y \lambda_{v_y}
\right)
\end{bmatrix}
$$

, allowing for instance method `EOM_Fullstate` to be defined as

```python
def EOM_Fullstate(self, fullstate, control):
    x, y, vx, vy, m, lx, ly, lvx, lvy, lm = fullstate
    u, ux, uy     = control
    T, Isp, g0, g = self.T, self.Isp, self.g0, self.g
    x0            = T*u/m
    x1            = T*u/m**2
    return array([
        [                  vx],
        [                  vy],
        [               ux*x0],
        [           uy*x0 - g],
        [       -T*u/(Isp*g0)],
        [                   0],
        [                   0],
        [                 -lx],
        [                 -ly],
        [x1*(lvx*ux + lvy*uy)]
    ], float)
```

Additionally, for increased performance of optimisation and numerical integration one can also define the *Jacobian* matrix of the full state equations of motion

$$
\frac{\partial (\dot{\pmb{s}} \cup \dot{\pmb{\lambda}})}{\partial (\pmb{s} \cup \pmb{\lambda})}
=
\begin{bmatrix}
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & - \frac{T \hat{u}_x}{m^{2}} u & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & - \frac{T \hat{u}_y}{m^{2}} u & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & -1 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & -1 & 0 & 0 & 0\\0 & 0 & 0 & 0 &
- \frac{2 T u}{m^{3}}
\left(
  \hat{u}_y \lambda_{v_y} +  \lambda_{v_x} \hat{u}_x
\right)
& 0 & 0 & \frac{T \hat{u}_x}{m^{2}} u & \frac{T \hat{u}_y}{m^{2}} u & 0
\end{bmatrix}
$$

which manifests itself in the instance method

```python
def EOM_Fullstate_Jac(self, fullstate, control):
    x, y, vx, vy, m, lx, ly, lvx, lvy, lm = fullstate
    u, ux, uy = control
    T         = self.T
    x0        = T*u/m**2
    x1        = ux*x0
    x2        = uy*x0
    x3        = 2*T*u/m**3
    return array([
        [0, 0, 1, 0,                      0,  0,  0,  0,  0, 0],
        [0, 0, 0, 1,                      0,  0,  0,  0,  0, 0],
        [0, 0, 0, 0,                    -x1,  0,  0,  0,  0, 0],
        [0, 0, 0, 0,                    -x2,  0,  0,  0,  0, 0],
        [0, 0, 0, 0,                      0,  0,  0,  0,  0, 0],
        [0, 0, 0, 0,                      0,  0,  0,  0,  0, 0],
        [0, 0, 0, 0,                      0,  0,  0,  0,  0, 0],
        [0, 0, 0, 0,                      0, -1,  0,  0,  0, 0],
        [0, 0, 0, 0,                      0,  0, -1,  0,  0, 0],
        [0, 0, 0, 0, -uy*lvy*x3 - lvx*ux*x3,  0,  0, x1, x2, 0]
    ], float)
```

The only necessary step left to be taken in order to implement *indirect methods* of trajectory optimisation is to define the instance method `Pontryagin`, named after [*Pontryagin's maximum principle*](https://en.wikipedia.org/wiki/Pontryagin's_maximum_principle), which takes as its parametres the system's full state $$ \pmb{s} \cup \pmb{\lambda} $$. Through maximizing the *Hamiltonian* over the set of all possible controls

$$
H(\pmb{s}^\star_k,\pmb{c}^\star_k,\lambda^\star_k)\leq H(\pmb{s}^\star_k,\pmb{c},\lambda^\star_k)~\forall \pmb{c} \in \mathcal{U}
$$

, an on board *optimal* control policy is found that maps the system's full state to an appropriate control, i.e. $$ \pmb{s}_k \mapsto \pmb{c}_k \forall k $$

In the specific case of the planetary lander being discussed, this manifests in the optimal thrust direction being

$$
\begin{gather}
\hat{u}_x = - \frac{\lambda_{v_x}}{\sqrt{\lambda_{v_x}^2 + \lambda_{v_y}^2}} \\
\hat{u}_y = - \frac{\lambda_{v_y}}{\sqrt{\lambda_{v_x}^2 + \lambda_{v_y}^2}}
\end{gather}
$$

and the optimal thrust throttle being

$$
u =
\left\{
\begin{align}
1 \text{ if } S < 0\\
0 \text{ if } S > 0
\end{align}
\right.
$$

, where the *switching function* is

$$ S = \frac{I_{sp} g_0 \sqrt{\lambda_{v_x}^2 + \lambda_{v_y}^2}}{m} - \lambda_m $$

After all the analytical derivations are done, one can define the instance method `Pontryagin` as

```python
def Pontryagin(self, fullstate):
    x, y, vx, vy, m, lx, ly, lvx, lvy, lm = fullstate
    lv = sqrt(abs(lvx)**2 + abs(lvy)**2)
    ux = -lvx/lv
    uy = -lvy/lv
    S  = 1 - self.Isp*self.g0*lv/m - lm
    if S < 0:
      u = 1
    elif S => 0:
      u = 0
    return u, ux, uy
```

and the full process of defining one's own `Dynamical_Model` is completed! One could now use any of the *Astro.IQ* methods of optimisation to optimise the trajectory.
