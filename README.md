# ISE AI Final Project: Collective Transport using Swarm Robotics


What should the main file do?
import all the dependacy for the simulation to work
- TODO: Localised all them first 
- For now, there's a known map
- Each robot, will init by broadcasting the id of each
- Object detection and roam around. RANDOM. (Obj is already in map not detected)
- One Robot will detec the object. -> 1 robot ask to be the master, if everybody agrees, taskmaster will assigned each "path_finding" for *all* of the robots.
- Program ends at all robots at the object


Spilt the runablefile
1. SLAM: 1 Turtlebot localising the same arena field
1. Consenses: 
1. Comms:
1. Object Detection:


# Creating the right environment (faster)
1. Using the following to the commands
```
conda create --name "swarm" python==3.12.7
conda activate swarm
pip install uv 
uv pip install -r requirements.txt
```
2. run `conda env create -f environment.yml` (simpler)

# Adding python path to webots
```where python``` to get the path then paste the `envs/swarm/bin/python` into the webots settings



<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
