Razer Hydra Teleop
-----------------------
[![Example video](https://img.youtube.com/vi/FDUHbQNJA5Q/0.jpg)](https://www.youtube.com/watch?v=FDUHbQNJA5Q)

ROS package that uses a [Razer Hydra](https://support.razer.com/console/razer-hydra/) to manipulate objects in a Drake simulation. Acquires signals from the controller via ROS messages from [aleeper/razer_hydra](https://github.com/aleeper/razer_hydra), and visualizes / allows controls of simulation through [Meshcat](https://github.com/rdeits/meshcat-python/).

# Usage

## Setup

Install [Drake](https://drake.mit.edu/) with Python bindings. A binary install should work fine. (TODO: Peg a version.)

Set up a version of ROS that's happy with Python3. I'm on a weird old setup that uses Melodic but uses [this guide](https://dhanoopbhaskar.com/blog/2020-05-07-working-with-python-3-in-ros-kinetic-or-melodic/) to get Python3 support; using something more modern should hopefully work the same.

In a ROS catkin workspace, clone
- this repo
- [aleeper/razer_hydra](https://github.com/aleeper/razer_hydra), the razer-hydra driver
and run `catkin_make`.

## Run

In one terminal:
- `roslaunch razer_hydra hydra.launch`


