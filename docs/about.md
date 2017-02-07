---
title: About
permalink: /Astro.IQ/about
---

Machine Learning<sup id="a1">(1)</sup> + Spacecraft Trajectory Optimisation<sup id="a2">(2)</sup>

## Inspiration
A housefly is a rather simple organism, yet it is able to independently make decisions to achieve its goals, such as navigating to a food-source and avoiding obstacles. Inspecting closer, a housefly is able to make these decisions instantaneously, such as in the case of being swatted at by a human. If one thinks about the descent of a lander onto the Martian surface, the nature of the situation is quite the same. Because communication with Earth is prolonged, the lander must make decisions on its own in order to safely land on the surface. If a common housefly can independently make decisions in real-time, in uncertain dynamic environments, than surely a spacecraft should be able to do the same in an environment where the objective is clearly outlined.

## Goal
This library aims to implement various *machine learning* and *computational intelligence* techniques in a variety of common *astrodynamics* applications.

## Applications
- [ ] Planetary Landings
- [ ] Cislunar Trajectories
- [ ] Interplanetary Trajectories

## Trajectory Optimisation Architectures
- [ ] High Fidelity Low-Thrust Direct Transcription <b id="r1">(Yam et al.)</b>
- [ ] Sims and Flanagan Direct Transcription <b id="r2">(Sims et al.)</b>

## Machine Learning Architectures
- [ ] Shallow Feed-Forward Networks
- [ ] Deep Feed-Forward Networks
- [ ] Recurrent Networks
- [ ] Deep-Q Learning <b id="ml1">(Mnih et al.)</b>
- [ ] Guided Policy Search <b id="ml2">(Levine et al.)</b>
- [ ] Evolutionary Neurocontrol <b id="ml3">(Dachwald)</b>


#### Definitions
<b id="f1">(1)</b> A type of artificial intelligence (AI) that provides computers with the ability to learn without being explicitly programmed.  
<b id="f2">(2)</b> An especially complicated continuous optimisation problem, which is characterised by: 1) nonlinear dynamics, 2) many practical trajectories and state variable discontinuities, 3) inexplicit terminal conditions (e.g. departure and arrival planet positions), 4) time-dependant influences (i.e. from planet positions determined from ephemerides), 5) apriori unknown basic optimal trajectory structure.  

#### References
 <b id="r1">(Yam et al.)</b> Yam, C. H., Izzo, D., & Biscani, F. (2010). Towards a High Fidelity Direct Transcription Method for Optimisation of Low-Thrust Trajectories. 4th International Conference on Astrodynamics Tools and Techniques, 1–7. Retrieved from http://arxiv.org/abs/1004.4539  
<b id="r2">(Sims et al.)</b> Sims, J. A., & Flanagan, S. N. (1997). Preliminary Design of Low-Thrust Interplanetary Missions.  
<b id="ml1">(Mnih et al.)</b> Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Petersen, S. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.  
<b id="ml2">(Levine et al.)</b> Levine, S., & Koltun, V. (2013). Guided Policy Search. In ICML (3) (pp. 1-9).  
<b id="ml3">(Dachwald)</b> Dachwald, B. (2005). Optimization of very-low-thrust trajectories using evolutionary neurocontrol. Acta Astronautica, 57(2), 175-185.  
