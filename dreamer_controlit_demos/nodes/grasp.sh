#!/bin/bash

# USAGE: rosrun dreamer_controlit_demos peggy_right_x.py <position x> <y> <z> <orientation w> <x> <y> <z>

# use default postion and orientation
rosrun dreamer_controlit_demos peggy_right_10.py 0.3080369181313972 -0.0933579900905777 1.01059794106796 -0.33079164055278293 0.7242129795026276 0.108525588846752 0.595243351433504

# UNCOMMENT below for: postion doesn't change only orientation
#rosrun dreamer_controlit_demos peggy_right_x.py 0.3080369181313972 -0.0933579900905777 1.01059794106796

# UNCOMMENT below for: position change, orientation doesn't change
#rosrun dreamer_controlit_demos peggy_right_x.py $* -0.33079164055278293 0.7242129795026276 0.108525588846752 0.595243351433504
