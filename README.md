# i-SPin: Multicomponent Schrodinger-Poisson systems with self-interactions

A code to simulate 3-component Schrodinger systems (with/without gravity + with/without self-interactions) obeying SO(3) symmetry

based on arXiv:2211.08433 with Mustafa A. Amin

The main file to run is i-SPin_main.py. It allows the user to input desired parameters, along with initial conditions for the Schrodinger fields. 
It uses three function modules: 

(1) "kdkfunc.py" which performs the main (drift-kick-drift) steps of the algorithm; 

(2) "conservefunc.py" which tracks the fractional changes in total mass and spin in the system; 

(3) "rtplotfunc.py" which is a module to plot different things. The default setting is to plot mass and spin density projections along with total mass and spin fractional changes.

(Python modules available. Mathematica modules coming soon...)
