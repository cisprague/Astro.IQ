---
title: Tutorials
permalink: /tutorials/
---

## It's as easy as this
```python
#Import the necessary modules
from context import Massive_Body
#Instantiate Earth as a massive celestial object
Earth = Celestial_Body('Earth')
#Instantiate Mars as a massive celestial object
Mars = Celestial_Body('Mars')
#Instantiate the debris fragment as an orbital body
Sat = Earth.Satellites.Fengyun_1C.Fengyun_1C_Deb_102
#Times at which to compute position and velocity
times = [2457061.5, 2457062.5, 2457063.5, 2457064.5]
#Compute the position and velocity of the debris
#fragment with respect to the centre of Mars.
p, v = Sat.Position_and_Velocity_WRT(Mars, times[0])

#Show results
print('The position [km] and velocity [km/s] of')
print('Fengyun_1C_Deb_102 with respect to the centre of Mars:')
print('Position:'), p * 1e-3
print('Velocity: '), v * 1e-3
```
```
The position [km] and velocity [km/s] of
Fengyun_1C_Deb_102 with respect to the centre of Mars:
Position: [ -3.16064398e+08   4.67978840e+07   2.47605342e+07]
Velocity:  [-15.22992539 -49.26948181 -14.59125297]
```
