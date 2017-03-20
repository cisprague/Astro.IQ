---
title: "Generating a Data Base with Homotopic Trajectory Transitioning"
author: "Christopher Iliffe Sprague"
---

```python
# We first import the resources
import sys
sys.path.append('..')
from Trajectory import Point_Lander
from Optimisation import Hermite_Simpson
from PyGMO import *
from numpy import *
from pandas import *
import matplotlib.pyplot as plt
```


```python
# We instantiate the problem
prob = Hermite_Simpson(Point_Lander(si=[0,1000,20,-5,9900]))
```


```python
# Let us first supply a ballistic trajectory as a guess
zguess = prob.Guess.Ballistic(tf=28)
```


```python
# Decode the guess so we can visualise
tf, cb, s, c = prob.Decode(zguess)
```


```python
plt.plot(s[:,0], s[:,1], 'k.-')
plt.axes().set_aspect('equal', 'datalim')
plt.xlabel('Cross Range [m]')
plt.ylabel('Altitude [m]')
plt.show()
```


![png]({{ site.baseurl }}/assets/Generating%20Data_files/Generating%20Data_5_0.png)



```python
# Use SLSQP and alternatively Monotonic Basin Hopping
alg1 = algorithm.scipy_slsqp(max_iter=3000, screen_output=True)
alg2 = algorithm.mbh(alg1, screen_output=True)
```


```python
# Instantiate a population with 1 individual
pop = population(prob)
# and add the guess
pop.push_back(zguess)
```


```python
# We then optimise the trajectory
pop = alg1.evolve(pop)
```

      NIT    FC           OBJFUN            GNORM
        1   231    -9.900000E+03     1.000000E+00
        2   463    -9.910000E+03     1.000000E+00
        3   694    -9.910999E+03     1.000000E+00
        4   925    -9.912809E+03     1.000000E+00

     1311 304389    -9.626603E+03     1.000000E+00
     1312 304620    -9.626603E+03     1.000000E+00
     1313 304851    -9.626603E+03     1.000000E+00
     1314 305082    -9.626603E+03     1.000000E+00
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: -9626.60290038
                Iterations: 1314
                Function evaluations: 305083
                Gradient evaluations: 1314



```python
# We now visualise the optimised trajectory
tf, cb, s, c = prob.Decode(pop.champion.x)
plt.plot(s[:,0], s[:,1], 'k.-')
plt.axes().set_aspect('equal', 'datalim')
plt.xlabel('Cross Range [m]')
plt.ylabel('Altitude [m]')
plt.show()
```


![png]({{ site.baseurl }}/assets/Generating%20Data_files/Generating%20Data_9_0.png)



```python
# and look at the control throttle
plt.close('all')
f, ax = plt.subplots(2, sharex=True)
ax[0].plot(c[:,0], 'k.-')
ax[1].plot(cb[:,0], 'k.-')
plt.xlabel('Node Index')
ax[0].set_ylabel('Node Throttle')
ax[1].set_ylabel('Midpoint Throttle')
plt.show()
```


![png]({{ site.baseurl }}/assets/Generating%20Data_files/Generating%20Data_10_0.png)



```python
# We save the optimised trajectory decision vector
save('HSD0', pop.champion.x)
```


```python
# We perturb the initial state and find a optimsal trajectory
# Decreasing the velocity by a few m/s for demonstration
prob = Hermite_Simpson(Point_Lander(si=[0,1000,10,-5,9900]))
# and create a population for the new problem
pop = population(prob)
# and add the previous population decision vector
pop.push_back(z)
```


```python
# We then optimise the trajectory, this now will not take long!
pop = alg1.evolve(pop)
```

      NIT    FC           OBJFUN            GNORM
        1   231    -9.626603E+03     1.000000E+00
        2   462    -9.626055E+03     1.000000E+00
        3   693    -9.625759E+03     1.000000E+00
        4   924    -9.625883E+03     1.000000E+00

       56 12936    -9.660155E+03     1.000000E+00
       57 13167    -9.661138E+03     1.000000E+00
       58 13398    -9.661133E+03     1.000000E+00
       59 13629    -9.661133E+03     1.000000E+00
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: -9661.13264337
                Iterations: 59
                Function evaluations: 13629
                Gradient evaluations: 59



```python
# We now compare this new perturbed trajectory to the previous
tf1, cb1, s1, c1 = prob.Decode(pop.champion.x)
plt.close('all')
plt.figure()
plt.plot(s1[:,0], s1[:,1], 'k.-') # The new trajectory
plt.plot(s[:,0], s[:,1], 'k.--') # The old trajectory
plt.legend(['Old Trajectory', 'New Trajectory'])
plt.title('Homotopic Trajectory Transitioning')
plt.axes().set_aspect('equal', 'datalim')
plt.xlabel('Cross Range [m]')
plt.ylabel('Altitude [m]')
plt.show()
```


![png]({{ site.baseurl }}/assets/Generating%20Data_files/Generating%20Data_14_0.png)



```python
# In essence, we will incrementally perturb the dynamical system's
# initial state and repeatedly compute new optimal trajectories.
# So, again we store the new trajectory decisions vector
save('HSD1', pop.champion.x)
```


```python
# Store the current decision
z = pop.champion.x
```


```python
# Now we try perturbing the initial position rather
# We instantiate the problem
prob = Hermite_Simpson(Point_Lander(si=[-10,1000,10,-5,9900]))
# and create a population for the new problem
pop = population(prob)
# and add the previous population decision vector
pop.push_back(z)
```


```python
# We then optimise the trajectory, this now will not take long!
pop = alg1.evolve(pop)
```

      NIT    FC           OBJFUN            GNORM
        1   231    -9.661133E+03     1.000000E+00
        2   462    -9.659388E+03     1.000000E+00
        3   693    -9.659415E+03     1.000000E+00
        4   924    -9.659430E+03     1.000000E+00

       19  4389    -9.660673E+03     1.000000E+00
       20  4620    -9.660756E+03     1.000000E+00
       21  4851    -9.661134E+03     1.000000E+00
       22  5082    -9.661133E+03     1.000000E+00
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: -9661.13264337
                Iterations: 22
                Function evaluations: 5083
                Gradient evaluations: 22



```python
# We now compare this new perturbed trajectory to the previous
tf2, cb2, s2, c2 = prob.Decode(pop.champion.x)
```


```python
plt.close('all')
plt.figure()
plt.plot(s2[:,0], s2[:,1], 'k.-') # The new trajectory
plt.plot(s1[:,0], s1[:,1], 'k.--') # The old trajectory
plt.plot(s[:,0], s[:,1], 'k.--') # The initial trajectory
plt.legend(['New', 'Old', 'Initial'])
plt.title('Homotopic Trajectory Transitioning')
plt.axes().set_aspect('equal', 'datalim')
plt.xlabel('Cross Range [m]')
plt.ylabel('Altitude [m]')
plt.show()
```


![png]({{ site.baseurl }}/assets/Generating%20Data_files/Generating%20Data_20_0.png)



```python
# Again, save the decision
save('HSD2', pop.champion.x)
z = pop.champion.x
```


```python
# We instantiate the problem one last time
prob = Hermite_Simpson(Point_Lander(si=[-10,1000,0,-5,9900]))
# and create a population for the new problem
pop = population(prob)
# and add the previous population decision vector
pop.push_back(z)
```


```python
# We then optimise the trajectory, this now will not take long!
pop = alg1.evolve(pop)
```

      NIT    FC           OBJFUN            GNORM
        1   231    -9.661133E+03     1.000000E+00
        2   462    -9.626934E+03     1.000000E+00
        3   693    -9.626588E+03     1.000000E+00
        4   924    -9.626690E+03     1.000000E+00

      115 26575    -9.660552E+03     1.000000E+00
      116 26806    -9.660562E+03     1.000000E+00
      117 27037    -9.660590E+03     1.000000E+00
      118 27268    -9.660590E+03     1.000000E+00
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: -9660.58977623
                Iterations: 118
                Function evaluations: 27269
                Gradient evaluations: 118



```python
# We now compare this new perturbed trajectory to the previous
tf3, cb3, s3, c3 = prob.Decode(pop.champion.x)
```


```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
plt.close('all')
plt.figure()
plt.plot(s3[:,0], s2[:,1], 'k.-')
plt.plot(s2[:,0], s2[:,1], 'k.--') # The new trajectory
plt.plot(s1[:,0], s1[:,1], 'k.--') # The old trajectory
plt.plot(s[:,0], s[:,1], 'k.--') # The initial trajectory
plt.legend(['New', 'Old'])
plt.title('Homotopic Trajectory Transitioning')
plt.axes().set_aspect('equal', 'datalim')
plt.xlabel('Cross Range [m]')
plt.ylabel('Altitude [m]')
#plt.savefig('Homotopic_Transitioning.pdf', format='pdf',
#           transparent=True, bbox_inches='tight')
plt.show()
```


![svg]({{ site.baseurl }}/assets/Generating%20Data_files/Generating%20Data_25_0.svg)



```python
# I think we get it now... time to do this more programmatically
# but first we again save the decision vector
z = pop.champion.x
save('HSD3', z)
```
