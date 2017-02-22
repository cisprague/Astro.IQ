---
title: "Indirect Trajectory Optimization for Cislunar Trajectories"
author: "Christopher Iliffe Sprague"
---

## Dynamics
The dynamics of a spacecraft in the *circular restricted three-body problem* (CRTBP) are characterised in scaler form by the ordinary differential equations

$$
\begin{align}
  \dot{x} &= v_x \\
  \dot{y} &= v_y \\
  \dot{z} &= v_z \\
  \dot{v}_x &= x - \frac{(1-\mu)(x+\mu)}{r_1^3} - \frac{\mu (x+\mu-1)}{r_2^3} + 2 v_y + \frac{u T \hat{u}_x}{m} \\
  \dot{v}_y &= y - \frac{(1-\mu)y}{r_1^3}
  - \frac{\mu y}{r_2^3} - 2 v_x + \frac{u T \hat{u}_y}{m} \\
  \dot{v}_z &= - \frac{(1-\mu)z}{r_1^3}
  - \frac{\mu z}{r_2^3} + \frac{u T \hat{u}_z}{m} \\
  \dot{m} &= -\frac{u T}{I_{sp} g_0} \\
  r_1 &= \left((x+\mu)^2+y^2+z^2\right)^{1/2} \\
  r_2 &= \left((x+\mu-1)^2+y^2+z^2\right)^{1/2}
\end{align}
$$

where $$[x,y,z,v_x,v_y,v_z,m]^\intercal$$ describes the state of the spacecraft and $$u [\hat{u}_x,\hat{u}_y,\hat{u}_z]^\intercal$$ is the control to be chosen along the spacecraft's trajectory, where the throttle $$u \in [0,1]$$ and thrust direction $$\sqrt{\hat{u}_x^2 + \hat{u}_y^2 + \hat{u}_z^2} = 1$$. The parameters inherent to the problem are:

1. Maximum thrust: $$T$$
2. Specific impulse: $$I_{sp}$$
3. Earth's sea-level gravity: $$g_0$$

## Cost Function
The desire for most spacecraft trajectories is to reduce fuel consumption. Such as it is, an *optimal trajectory* should minimize the *homotopic* path cost

$$
\mathcal{J} = \frac{T}{I_{sp} g_0} \int_{t_0}^{t_f} (u - \alpha u (1-\alpha)) dt
$$

from the initial time $$t_i$$ to the *specified* final time $$t_f$$. The parameter $$\alpha$$ is implemented in order to avoid numerical convergence difficulties associated with the discontinuous nature of mass optimal *bang-bang* control. In practice, the trajectory is initially optimized with the homotopy parameter $$\alpha = 1$$ for convergence ease, after which the trajectory is optimized for iteratively smaller values until $$\alpha=0$$, corresponding to a mass path cost.

## Optimal Control
The *dynamical system* is a *Hamiltonian System*, and its *Hamiltonian* is

$$
\begin{multline}
\mathcal{H} = \lambda_x v_x + \lambda_y v_y + \lambda_z v_z \\
+ \lambda_{v_x} \left( x - \frac{(1-\mu)(x+\mu)}{r_1^3} - \frac{\mu (x+\mu-1)}{r_2^3} + 2 v_y + \frac{u T \hat{u}_x}{m} \right) \\
+ \lambda_{v_y} \left(y - \frac{(1-\mu)y}{r_1^3}
- \frac{\mu y}{r_2^3} - 2 v_x + \frac{u T \hat{u}_y}{m} \right) \\
+ \lambda_{v_z} \left(\frac{(1-\mu)z}{r_1^3}
- \frac{\mu z}{r_2^3} + \frac{u T \hat{u}_z}{m} \right)
- \lambda_m \frac{u T}{I_{sp} g_0} \\
+ \frac{T}{I_{sp} g_0}  (u - \alpha u (1-\alpha))
\end{multline}
$$

## Nonlinear Parameter Optimization
Determine the decision vector

$$ [\lambda_x, \lambda_y, \lambda_z, \lambda_{v_x}, \lambda_{v_y},
\lambda_{v_z}, \lambda_m] \Bigr\rvert_{t_0} $$
