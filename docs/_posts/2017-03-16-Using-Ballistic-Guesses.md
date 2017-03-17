---
title: "Using Ballistic Guesses"
author: "Christopher Iliffe Sprague"
---

In the process of *trajectory optimisation*, often the most severe bottleneck in performance manifests in supplying an *initial guess*. One of the simplest ways to supply an initial guess is to initialise the *nonlinear programme decision vector* $$\pmb{Z}$$ within the *state* and *control* space, $$\mathcal{S}$$ and $$\mathcal{U}$$ respectively. At first this may seem like a sufficient idea; however, if one brings their attention to the problem's dynamic constraints, this will certainly not work.

Examining a two dimensional planetary lander, whose dynamics are governed by the first order system ordinary differential equations

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
$$
\begin{align}
\text{Position: } & \pmb{r} = [x, y] \\
\text{Velocity: } & \pmb{v} = [v_x, v_y] \\
\text{Mass: } & m = 9500~kg \\
\text{Maximum Thrust: } & T = 44000~N \\
\text{Throttle: } & u \in [0, 1] \\
\text{Thrust Direction: } & \hat{\pmb{u}} = [\hat{u}_x, \hat{u}_y] \in [-1, 1] \\
\text{Moon Gravity: } & g = 1.6229~m/s^2 \\
\text{Earth Gravity: } & g_0 = 9.802~m/s^2 \\
\text{Specific Impulse: } & I_{sp} = 311~s
\end{align}
$$

, it is quite unsurprising that a random distribution of state nodes is likely to prevent the optimiser from converging to a *feasible* solutions. A smarter way to provide a guess that looks more natural, with respect to the system's dynamics, is to provide a *ballistic guess*, that is an uncontrolled trajectory.

To do this one simply *propagates* or numerically integrates the system's dynamics from an initial state to an arbitrary *final time* $$t_f$$, and then samples $$M$$ evenly distributed nodes along the trajectory to subsequently input to the optimiser. However, one must ensure that all these nodes of the system's state are within the problem's state space bounds, that is $$\pmb{s}_k \forall k \in \{1,\dots,M\} \subset \mathcal{S} $$.

## Astro.IQ Implementation
Within the [*Astro.IQ*](https://github.com/CISprague/Astro.IQ) framework, one can easily define their own dynamical model, choose an optimisation method, and subsequently solve the trajectory optimisation with any of the many algorithms of [*PyGMO*](https://github.com/esa/pagmo). One can employ this methodology through the following step:

One first imports the necessary resources

```python
from Trajectory   import Point_Lander
from Optimisation import Trapezoidal
```

then instantiates the dynamical model.

```python
# Instantiate a dynamical model and look at details
Model = Point_Lander()
```

### Optimisation Transcription
The problem is then transcribed with a chosen optimisation method. In this case the *trapezoidal transcription* method is used, where the *decision vector* is

$$
\pmb{Z}^\intercal = [\pmb{s}_1, \pmb{c}_1, \dots, \pmb{s}_M, \pmb{c}_M]
$$

, where the system's *state* is given by

$$ \pmb{s}_k^\intercal = {[x,y,v_x,v_y,m]}_k \in \mathcal{S} $$

and the *control* by

$$ \pmb{c}_k^\intercal = {[u, \hat{u}_x, \hat{u}_u]}_k \in \mathcal{U}$$

. In this transcription, the *dynamic constraints* are enforced by *equality constraints* given by the trapezoidal quadrature

$$
\pmb{\zeta}_k = \pmb{s}_{k+1} - \pmb{s}_k - \frac{t_{k+1} - t_k}{2}(\dot{\pmb{s}}_k + \dot{\pmb{s}}_{k+1}) \forall k \in [1, N]
$$

, where $$N=M-1$$ is the number of segments the trajectory is divided into.

```python
# Create a trajectory optimisation problem and look at details
Problem = Trapezoidal(Model, nsegs=50)
```

### Taking a Guess
One can show what a *ballistic trajectory* with a $$t_f = 20~s$$ flight time looks like, where the problem's number of nodes is implicitly passed to the guessing method:

```python
# Guess from a ballistic (uncontrolled) trajectory
tf, state, control = Problem.Guess.Ballistic(tf=20, nlp=False)
```

```python
# Visualise the guess
import matplotlib.pyplot as plt
plt.plot(state[:,0], state[:,1], 'k.-') # Trajectory
plt.ylabel('Altitude [m]'); plt.xlabel('Cross Range [m]')
plt.show()
```


![png]({{ site.baseurl }}/assets/Guessing_files/Guessing_5_0.png)

### Optimising the Trajectory

The optimiser is imported

```python
# Import PyGMO for optimisation
from PyGMO import *
```

wherein the method of [*sequential least squares quadratic programming*](https://en.wikipedia.org/wiki/Sequential_quadratic_programming) is used:

```python
# Use sequential least squares quadratic programming
algo = algorithm.scipy_slsqp(max_iter=3000, screen_output=True)
```

A space for any number of *decision* vectors is allotted:

```python
# Create an empty population space for individuals (decision vectors) to inhabit
pop = population(Problem)
```

A *decision vector* $$\pmb{Z}$$ is generated

```python
# Provide a ballistic (uncontrolled) trajectory as an initial guess
zguess = Problem.Guess.Ballistic(tf=20)
```

, and subsequently added as a *chromosome* to the space.

```python
# Add the guess to the population
pop.push_back(zguess)
```

The *decision vector* is the evolved through the *gradient based* optimizer until an error tolerance is satisfied, both with respect to the *boundary conditions* and *dynamic constraints*.

```python
# Evolve the individual with SLSQP
pop = algo.evolve(pop)
```

      NIT    FC           OBJFUN            GNORM
        1   411    -9.500000E+03     1.000000E+00
        2   832    -9.500000E+03     1.000000E+00
        3  1250    -9.500000E+03     1.000000E+00
        4  1671    -9.500000E+03     1.000000E+00
        ⋮   ⋮           ⋮                ⋮
      627 257760    -9.242340E+03     1.000000E+00
      628 258175    -9.242340E+03     1.000000E+00
      629 258590    -9.242340E+03     1.000000E+00
      630 259005    -9.242340E+03     1.000000E+00
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: -9242.33995699
                Iterations: 630
                Function evaluations: 259006
                Gradient evaluations: 629

The optimised *decision vector* is then decoded into the total flight time $$t_f$$, the state trajectory $$\pmb{s}_k \forall k$$, and the sequence of controls $$\pmb{c}_k \forall k$$.

```python
tf, s, c = Problem.Decode(pop.champion.x)
```

### Trajectory Evaluation

Examining the plot of the optimised position trajectory, the optimisation process seems to have been quite effective.

```python
# x vs. y
plt.plot(s[:,0], s[:,1], 'k.-'); plt.xlabel('Cross range [m]'); plt.ylabel('Altitude [m]')
plt.show()
```


![png]({{ site.baseurl }}/assets/Guessing_files/Guessing_13_0.png)

#### Control
From *optimal control* theory, one can analyse how the nature of the system's control should behave. One introduces a vector of non physical *costate* variables

$$ \pmb{\lambda}^\intercal = [\lambda_x, \lambda_y, \lambda_{v_x}, \lambda_{v_y}, \lambda_m] $$

and subsequently defines the system's *Hamiltonian*

$$ \mathcal{H} = \pmb{\lambda}^\intercal \dot{\pmb{s}} + \mathcal{L} $$

, where the system's *Lagrangian* or *cost functional* is defined as

$$
\mathcal{L} = T u
$$

. From [*Pontryagin's maximum principle*](https://en.wikipedia.org/wiki/Pontryagin's_maximum_principle), which requires that the *Hamiltonian* must be maximized over the set of all possible controls $$\mathcal{U}$$

$$
H(\pmb{s}^\star_k,\pmb{c}^\star_k,\lambda^\star_k)\leq H(\pmb{s}^\star_k,\pmb{c},\lambda^\star_k)~\forall \pmb{c} \in \mathcal{U}
$$

, one finds that optimal throttle is

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

. Hence this describes what is known as *bang-bang control*, which is characteristic of *mass-optimal control*, where the throttle is either on or off. Plotting the throttle sequence of the planetary lander in this example, it can be seen that this nature is followed, with the exception of some intermediate values due to the problem's discretisation.


```python
# Controls (note the bang-off-bang profile indicative of mass optimal control)
plt.plot(c[:,0], 'k.-')
plt.ylabel('Throttle $u \in [0,1]$')
plt.xlabel('Node Index')
plt.show()
```


![png]({{ site.baseurl }}/assets/Guessing_files/Guessing_14_0.png)

The fuel expenditure is the plotted:

```python
# Plot the propellent usage
plt.plot(s[:,4],'k.-')
plt.ylabel('Mass [kg]')
plt.xlabel('Node Index')
plt.show()
```


![png]({{ site.baseurl }}/assets/Guessing_files/Guessing_15_0.png)

#### Soft Landing?

The lander not only met its target position, but it also landed softly as specified.

```python
# Soft landing
plt.plot(s[:,2], 'k.-')
plt.plot(s[:,3], 'k.--')
plt.legend(['$v_x$', '$v_y$'])
plt.xlabel('Node Index')
plt.ylabel('Velocity [m/s]')
plt.show()
```


![png]({{ site.baseurl }}/assets/Guessing_files/Guessing_16_0.png)


## Source Code
Have a look at the fill IPython notebook [here](https://github.com/CISprague/Astro.IQ/blob/master/src/Notebook/Guessing.ipynb)
