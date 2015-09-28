#!/usr/bin/env python

'''
Enables users to telemanipulate Dreamer's end effectors via CARL.
Uses WBOSC configured with due end-effector 5DOF control.

-----------------
Dependency notes:

If you're using Python 2.7, you need to install Python's enum package.

Download it from here: https://pypi.python.org/pypi/enum34#downloads

Then run:
  $ sudo python setup.py install

Visualzing the FSM requires smach_viewer:
  $ sudo apt-get install ros-indigo-smach-viewer

You will need to modify /opt/ros/indigo/lib/python2.7/dist-packages/xdot/xdot.py
lines 487, 488, 593, and 594 to contain self.read_float() instead of self.read_number().

-----------------
Usage Notes:

To issue a command using the command line:

  exit:
    $ rostopic pub --once /demo9/cmd std_msgs/Int32 'data: 0'

  go to ready:
    $ rostopic pub --once /demo9/cmd std_msgs/Int32 'data: 1'

  go to idle:
    $ rostopic pub --once /demo9/cmd std_msgs/Int32 'data: 2'

To visualize FSM:
  $ rosrun smach_viewer smach_viewer.py
'''

import sys, getopt     # for getting and parsing command line arguments
import time
import math
import threading
import rospy
import smach
import smach_ros

from enum import IntEnum

from std_msgs.msg import Float64, Float64MultiArray, MultiArrayDimension, Bool, Int32

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print(sys.path)


import DreamerInterface
import Trajectory
import TrajectoryGeneratorCubicSpline

# import roslib; roslib.load_manifest('controlit_dreamer_integration')

import controlit_trajectory_generators  # This adds the directory containing TrapezoidVelocityTrajGen to the PYTHONPATH
import TrapezoidVelocityTrajGen

# The previous demos that we would like to execute
import Demo4_HandWave
import Demo5_HandShake
import Demo7_HookemHorns

ENABLE_USER_PROMPTS = False

# Shoulder abductors about 10 degrees away from body and elbows bent 90 degrees
# DEFAULT_POSTURE = [0.0, 0.0,                                    # torso
#                    0.0, 0.174532925, 0.0, 1.57, 0.0, 0.0, 0.0,  # left arm
#                    0.0, 0.174532925, 0.0, 1.57, 0.0, 0.0, 0.0]  # right arm

# Shoulder abductors and elbows at about 10 degrees
DEFAULT_POSTURE = [0.0, 0.0,                                    # torso
                   0.0, 0.174532925, 0.0, 0.174532925, 0.0, 0.0, 0.0,  # left arm
                   0.0, 0.174532925, 0.0, 0.174532925, 0.0, 0.0, 0.0]  # right arm

DEFAULT_READY_RH_CARTPOS = [0.25822435038901964, -0.1895604971725577, 1.0461857180093073]
DEFAULT_READY_RH_ORIENT = [0.5409881394605172, -0.8191390472602035, 0.19063854336595773]
DEFAULT_READY_LH_CARTPOS = [0.25822435038901964, 0.25, 1.0461857180093073]
DEFAULT_READY_LH_ORIENT = [0.5409881394605172, 0.8191390472602035, 0.19063854336595773]
DEFAULT_READY_POSTURE = [0.06796522908004803, 0.06796522908004803,                                            # torso
    -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
    -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18]  # right arm


# Define the commands that can be received from the CARL user interface.
# These should match the commands defined in:
#
#     carl/carl_ws/src/carl_server/src/client/public/js/client.js
#
# See: https://bitbucket.org/cfok/carl
#
class Command:
    CMD_GOTO_IDLE = 0
    CMD_GOTO_READY = 1

    CMD_BEHAVIOR_SHAKE = 6
    CMD_BEHAVIOR_WAVE = 7
    CMD_BEHAVIOR_HOOKEM = 8
    CMD_BEHAVIOR_MORE = 9

    CMD_LEFT_GRIPPER = 1
    CMD_RIGHT_HAND = 2

    CMD_OPEN = 0
    CMD_CLOSE = 1

    CMD_TRANSLATE = 1
    CMD_ROTATE = 2

    CMD_MOVE_LEFT = 2
    CMD_MOVE_RIGHT = 3
    CMD_MOVE_UP = 4
    CMD_MOVE_DOWN = 5
    CMD_MOVE_FORWARD = 6
    CMD_MOVE_BACKWARD = 7

# Define the Cartesian directions. This is used by the MoveCartesianState.
class CartesianDirection(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    FORWARD = 4
    BACKWARD = 5

#==================================================================================
# The following parameters are used by the trapezoid velocity trajectory generator

# The time each trajectory should take
TIME_GO_TO_READY = 5.0
TIME_GO_TO_IDLE = 7.0

GO_BACK_TO_READY_SPEED = 0.01 # 1 cm/s

# The speed at which the Cartesian position should change
TRAVEL_SPEED = 0.02  # 2 cm per second
ACCELERATION = 0.06  # 1 cm/s^2
DECELERATION = 0.06  # 1 cm/s^2

# The update frequency when adjusting the Cartesian position
TRAJECTORY_UPDATE_FREQUENCY = 100

X_AXIS = 0
Y_AXIS = 1
Z_AXIS = 2

# Define the Cartesian movement increment
CARTESIAN_MOVE_DELTA = 0.01  # 1 cm movement increments


class TrajectoryState(smach.State):
    """
    A SMACH state that makes the robot follow a trajectory.
    """

    def __init__(self, dreamerInterface, traj):
        """
        The constructor.

        Keyword Parameters:
          - dreamerInterface: The object to which to provide the trajectory.
          - traj: The trajectory to follow.
        """

        smach.State.__init__(self, outcomes=["done", "exit"])
        self.dreamerInterface = dreamerInterface
        self.traj = traj

    def execute(self, userdata):
        rospy.loginfo('Executing TrajectoryState')

        if self.dreamerInterface.followTrajectory(self.traj):
            return "done"
        else:
            return "exit"

class TrajectoryShakeHands(smach.State):
    """
    A SMACH state that makes the robot shake hands.
    """

    def __init__(self, dreamerInterface, prevTraj):
        """
        The constructor.

        Keyword Parameters:
          - dreamerInterface: The object to which to provide the trajectory.
          - prevTraj: The previous trajectory. Used to ensure smooth transition into this trajectory
        """

        smach.State.__init__(self, outcomes=["done", "exit"])
        self.dreamerInterface = dreamerInterface

        TIME_EXTEND = 2.0
        TIME_SHAKE = 5.0

        # ==============================================================================================
        # Define the HandShake trajectory
        self.trajExtend = Trajectory.Trajectory("Extend", TIME_EXTEND)
        self.trajExtend.setPrevTraj(prevTraj)

        self.trajExtend.addRHCartWP([0.3104136932597554, -0.23958048524931236, 1.0288961306030007])
        self.trajExtend.addRHCartWP([0.3091653465539496, -0.26014726583647924, 1.0356396464404043])
        self.trajExtend.addRHCartWP([0.30972113171989446, -0.3134234808726032, 1.0461728164960131])
        self.trajExtend.addRHCartWP([0.3046856805137895, -0.3398003519073846, 1.0510229674144407])
        self.trajExtend.addRHCartWP([0.3173024098532336, -0.34219626411209203, 1.0644529426815315])

        self.trajExtend.addRHOrientWP([0.5004519685854716, -0.8593135617342791, 0.10548947700350815])
        self.trajExtend.addRHOrientWP([0.5004519685854716, -0.8593135617342791, 0.10548947700350815])
        self.trajExtend.addRHOrientWP([0.5004519685854716, -0.8593135617342791, 0.10548947700350815])
        self.trajExtend.addRHOrientWP([0.5004519685854716, -0.8593135617342791, 0.10548947700350815])
        self.trajExtend.addRHOrientWP([0.5004519685854716, -0.8593135617342791, 0.10548947700350815])

        self.trajExtend.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        self.trajExtend.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        self.trajExtend.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        self.trajExtend.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        self.trajExtend.addLHCartWP(DEFAULT_READY_LH_CARTPOS)

        self.trajExtend.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        self.trajExtend.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        self.trajExtend.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        self.trajExtend.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        self.trajExtend.addLHOrientWP(DEFAULT_READY_LH_ORIENT)

        self.trajExtend.addPostureWP([0.028444513491162594, 0.028444513491162594,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.11219453452059695, 0.16546402971632587, -0.11395882910199566, 1.3763986059723414, 1.6190270985068778, -0.37718957675753256, -0.19311388740064422])
        self.trajExtend.addPostureWP([0.027789879631802322, 0.027789879631802322,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.09982449826834842, 0.14084614270850673, -0.0058361512535315884, 1.4018309516255258, 1.556134219337181, -0.34496454788689784, -0.0927880951612021])
        self.trajExtend.addPostureWP([0.027935113357263137, 0.027935113357263137,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.13328856860260982, 0.07890268167854148, 0.2603418600235295, 1.400536877824891, 1.5541783835179697, -0.3488103334889514, -0.07865545836463488])
        self.trajExtend.addPostureWP([0.028047416195966585, 0.028047416195966585,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.14405158269023477, 0.08330404179761804, 0.35485388080251956, 1.3998793638319809, 1.5468160099625425, -0.3217818596171365, -0.08263738134922839])
        self.trajExtend.addPostureWP([0.028212084482967685, 0.028212084482967685,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.19493960990942347, 0.07496994081163051, 0.3741360648243361, 1.398699928093654, 1.4974573555944546, -0.31669671800133464, -0.08953743861272202])

        # self.trajExtend.addRHCartWP([0.318892510188593, -0.269491700424895, 1.0496129045379383])
        # self.trajExtend.addRHCartWP([0.4203674239044759, -0.2743572663427439, 1.093377210945474])
        # self.trajExtend.addRHCartWP([0.44428170484806473, -0.2702914994062119, 1.0997219240825615])

        # self.trajExtend.addRHOrientWP([0.019676850165412, -0.9919085050057322, -0.12542064927618982])
        # self.trajExtend.addRHOrientWP([0.019676850165412, -0.9919085050057322, -0.12542064927618982])
        # self.trajExtend.addRHOrientWP([0.019676850165412, -0.9919085050057322, -0.12542064927618982])

        # self.trajExtend.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        # self.trajExtend.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        # self.trajExtend.addLHCartWP(DEFAULT_READY_LH_CARTPOS)

        # self.trajExtend.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        # self.trajExtend.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        # self.trajExtend.addLHOrientWP(DEFAULT_READY_LH_ORIENT)

        # self.trajExtend.addPostureWP([0.06796522908004803, 0.06796522908004803,
        #     -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
        #     0.1295249978515601, 0.08931744864236359, 0.09266642932477151, 1.471017685835546, 1.623110228955735, -0.23047980576603305, -0.04256081682008954])
        # self.trajExtend.addPostureWP([0.06796522908004803, 0.06796522908004803,
        #     -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
        #     0.5125541454745224, 0.08615348913651269, 0.09537100473609225, 1.1596340637330818, 1.6376422708010987, -0.21181495926711674, -0.12099974279280543])
        # self.trajExtend.addPostureWP([0.06796522908004803, 0.06796522908004803,
        #     -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
        #     0.6089666791320509, 0.08793921769116066, 0.07082243411581694, 1.0425401499369535, 1.641068191538357, -0.20250645610308562, -0.10309210189561308])

        self.trajShake = Trajectory.Trajectory("Shake", TIME_SHAKE)
        self.trajShake.setPrevTraj(self.trajExtend)

        self.trajShake.addRHCartWP([0.31071689277058917, -0.3422655576028866, 1.0548678951005588])
        self.trajShake.addRHCartWP([0.32266611868312994, -0.34026301768744155, 1.0861769204139096])
        self.trajShake.addRHCartWP([0.3231493426756944, -0.3390478820635321, 1.1081103121891949])
        self.trajShake.addRHCartWP([0.32513302810765704, -0.3344754858243065, 1.0673023989448422])
        self.trajShake.addRHCartWP([0.31903382422753407, -0.332566106394895, 1.0298997494550077])
        self.trajShake.addRHCartWP([0.30422147281718565, -0.3296577294337501, 0.9925362962729714])
        self.trajShake.addRHCartWP([0.29349153147057855, -0.3256993083767356, 0.9738006291466748])
        self.trajShake.addRHCartWP([0.32224426461480954, -0.3310542688610496, 1.0318421917898017])
        self.trajShake.addRHCartWP([0.32810140838167967, -0.3326205381514441, 1.0563530115045692])
        self.trajShake.addRHCartWP([0.3281756475492571, -0.33051419331773485, 1.0674362367695813])
        self.trajShake.addRHCartWP([0.3173024098532336, -0.34219626411209203, 1.0644529426815315])

        self.trajShake.addRHOrientWP([-0.05732874991178004, -0.9953331349468206, 0.07762322404079096])
        self.trajShake.addRHOrientWP([-0.05732874991178004, -0.9953331349468206, 0.07762322404079096])
        self.trajShake.addRHOrientWP([-0.05732874991178004, -0.9953331349468206, 0.07762322404079096])
        self.trajShake.addRHOrientWP([-0.05732874991178004, -0.9953331349468206, 0.07762322404079096])
        self.trajShake.addRHOrientWP([-0.05732874991178004, -0.9953331349468206, 0.07762322404079096])
        self.trajShake.addRHOrientWP([-0.05732874991178004, -0.9953331349468206, 0.07762322404079096])
        self.trajShake.addRHOrientWP([-0.05732874991178004, -0.9953331349468206, 0.07762322404079096])
        self.trajShake.addRHOrientWP([-0.05732874991178004, -0.9953331349468206, 0.07762322404079096])
        self.trajShake.addRHOrientWP([-0.05732874991178004, -0.9953331349468206, 0.07762322404079096])
        self.trajShake.addRHOrientWP([-0.05732874991178004, -0.9953331349468206, 0.07762322404079096])
        self.trajShake.addRHOrientWP([0.5004519685854716, -0.8593135617342791, 0.10548947700350815])

        # self.trajShake.addRHCartWP([0.4198258452591858, -0.2677478824239589, 1.0718061874076472])
        # self.trajShake.addRHCartWP([0.41484901169199156, -0.2649553323213928, 1.1202064247617263])
        # self.trajShake.addRHCartWP([0.40680396278502484, -0.26216770388699995, 1.1612709079169703])
        # self.trajShake.addRHCartWP([0.3991130614391514, -0.26497263319522746, 1.095510121486765])
        # self.trajShake.addRHCartWP([0.39594068875228766, -0.26657084754076593, 1.0571390274765482])
        # self.trajShake.addRHCartWP([0.3814822807674513, -0.2667886087533124, 1.0057247643234024])
        # self.trajShake.addRHCartWP([0.4017023868441705, -0.265128103191222, 1.0593857775673534])
        # self.trajShake.addRHCartWP([0.40082050687162646, -0.25746103897624, 1.0747661053533655])
        # self.trajShake.addRHCartWP([0.44428170484806473, -0.2702914994062119, 1.0997219240825615])

        # self.trajShake.addRHOrientWP([0.08341952451728452, -0.9959228255616, -0.034335236342701525])
        # self.trajShake.addRHOrientWP([0.08341952451728452, -0.9959228255616, -0.034335236342701525])
        # self.trajShake.addRHOrientWP([0.08341952451728452, -0.9959228255616, -0.034335236342701525])
        # self.trajShake.addRHOrientWP([0.08341952451728452, -0.9959228255616, -0.034335236342701525])
        # self.trajShake.addRHOrientWP([0.08341952451728452, -0.9959228255616, -0.034335236342701525])
        # self.trajShake.addRHOrientWP([0.08341952451728452, -0.9959228255616, -0.034335236342701525])
        # self.trajShake.addRHOrientWP([0.08341952451728452, -0.9959228255616, -0.034335236342701525])
        # self.trajShake.addRHOrientWP([0.08341952451728452, -0.9959228255616, -0.034335236342701525])
        # self.trajShake.addRHOrientWP([0.019676850165412, -0.9919085050057322, -0.12542064927618982])

        self.trajShake.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        self.trajShake.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        self.trajShake.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        self.trajShake.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        self.trajShake.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        self.trajShake.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        self.trajShake.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        self.trajShake.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        self.trajShake.addLHCartWP(DEFAULT_READY_LH_CARTPOS)

        self.trajShake.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        self.trajShake.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        self.trajShake.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        self.trajShake.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        self.trajShake.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        self.trajShake.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        self.trajShake.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        self.trajShake.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        self.trajShake.addLHOrientWP(DEFAULT_READY_LH_ORIENT)

        self.trajShake.addPostureWP([0.028108547792328124, 0.028108547792328124,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.17228978956798197, 0.07434885404064914, 0.375333638022727, 1.3888092319639083, 1.5079201767999826, -0.31641347528653985, -0.1015319969687243])
        self.trajShake.addPostureWP([0.02807599142615882, 0.02807599142615882,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.21459152416842134, 0.07263057405113098, 0.37081140433698745, 1.4567639364718938, 1.5089499315784638, -0.3175519769988421, -0.09818059072886748])
        self.trajShake.addPostureWP([0.02804606887225163, 0.02804606887225163,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.22334628339985577, 0.07243726113745749, 0.36951308061996, 1.5267467563477293, 1.5118402015343135, -0.317372505037564, -0.09714620016070197])
        self.trajShake.addPostureWP([0.028378035288426473, 0.028378035288426473,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.2230578085565455, 0.04639868920316817, 0.37944787856090473, 1.3943789342038184, 1.5090758333232006, -0.31659570132874054, -0.09672596370783676])
        self.trajShake.addPostureWP([0.028413501795785365, 0.028413501795785365,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.20810354502069642, 0.04637090733175224, 0.3773909915962375, 1.2753670180837835, 1.491115101584284, -0.3139068866754734, -0.10009597100447003])
        self.trajShake.addPostureWP([0.02852194867687742, 0.02852194867687742,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.17942085807353006, 0.0458858888438078, 0.3761147209696891, 1.1696550882674261, 1.4878651941128596, -0.3136020058406554, -0.09769358534069474])
        self.trajShake.addPostureWP([0.028437878552613094, 0.028437878552613094,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.1585783300454735, 0.040323674861854536, 0.3743689535705127, 1.1239572239760118, 1.4880696754409228, -0.31544202969105134, -0.09728755843221447])
        self.trajShake.addPostureWP([0.027972524735509117, 0.027972524735509117,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.21766110353149037, 0.04462270263066805, 0.37394659601800545, 1.272177850314399, 1.4867124148544228, -0.3018197603232544, -0.11123418524790345])
        self.trajShake.addPostureWP([0.027902050140807648, 0.027902050140807648,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.23246574644808587, 0.04535144367015018, 0.37470091392591226, 1.3438226521225294, 1.4895806533796303, -0.3037351903496162, -0.10710217274190663])
        self.trajShake.addPostureWP([0.028037409299435538, 0.028037409299435538,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.2300209266385682, 0.04269931702175429, 0.36885256667640404, 1.3881511959199457, 1.4898744925776721, -0.3105485868832839, -0.10095552101380002])
        self.trajShake.addPostureWP([0.028212084482967685, 0.028212084482967685,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.19493960990942347, 0.07496994081163051, 0.3741360648243361, 1.398699928093654, 1.4974573555944546, -0.31669671800133464, -0.08953743861272202])

        # self.trajShake.addPostureWP([0.04865117748149428, 0.04865117748149428,
        #     -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
        #     0.5100539709627897, 0.08444133618154223, 0.06842049038822166, 1.0865054308261715, 1.591123720790246, -0.17134524797663087, -0.03921983926274288])
        # self.trajShake.addPostureWP([0.048455087756938124, 0.048455087756938124,
        #     -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
        #     0.4942045388316596, 0.08110721616121903, 0.07283713518166213, 1.2837574563842622, 1.622933835955884, -0.15935330922519564, -0.06632537681755694])
        # self.trajShake.addPostureWP([0.048683132506192064, 0.048683132506192064,
        #     -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
        #     0.4931165313576383, 0.08089760780276584, 0.07306006247276281, 1.4371450349948687, 1.6198106425605086, -0.16061865418933732, -0.052347172205017584])
        # self.trajShake.addPostureWP([0.04928089374095996, 0.04928089374095996,
        #     -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
        #     0.4264617218227655, 0.08111685363998597, 0.07342748439610863, 1.2893819307514431, 1.6156698378713175, -0.1831398958325058, -0.030989478268296036])
        # self.trajShake.addPostureWP([0.04918174114168457, 0.04918174114168457,
        #     -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
        #     0.41759012102573556, 0.08039014392299232, 0.07385486035484423, 1.1625594272940112, 1.6163543469614716, -0.22397284029544148, 0.015600315773426939])
        # self.trajShake.addPostureWP([0.049178267906286176, 0.049178267906286176,
        #     -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
        #     0.39538629458684865, 0.07758623329951554, 0.07282511186655695, 1.0039475478126376, 1.6096325786611048, -0.2737624128208012, 0.06710587019565711])
        # self.trajShake.addPostureWP([0.048639714765515187, 0.048639714765515187,
        #     -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
        #     0.4392419468414035, 0.07362002354671543, 0.07747369240655214, 1.1417047684357426, 1.664454017785462, -0.09729819384049211, -0.11543652673903287])
        # self.trajShake.addPostureWP([0.04897246986660818, 0.04897246986660818,
        #     -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
        #     0.4302665745526137, 0.05490628163960686, 0.07767423556103505, 1.2154233796497025, 1.6597838065134571, -0.0980911685170022, -0.11061983935651921])
        # self.trajShake.addPostureWP([0.06796522908004803, 0.06796522908004803,
        #     -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
        #     0.6089666791320509, 0.08793921769116066, 0.07082243411581694, 1.0425401499369535, 1.641068191538357, -0.20250645610308562, -0.10309210189561308])


        self.trajRetract = Trajectory.Trajectory("Retract", TIME_EXTEND)
        self.trajRetract.setPrevTraj(self.trajShake)

        self.trajRetract.addRHCartWP([0.3173024098532336, -0.34219626411209203, 1.0644529426815315])
        self.trajRetract.addRHCartWP([0.3046856805137895, -0.3398003519073846, 1.0510229674144407])
        self.trajRetract.addRHCartWP([0.30972113171989446, -0.3134234808726032, 1.0461728164960131])
        self.trajRetract.addRHCartWP([0.3091653465539496, -0.26014726583647924, 1.0356396464404043])
        self.trajRetract.addRHCartWP([0.3104136932597554, -0.23958048524931236, 1.0288961306030007])
        self.trajRetract.addRHCartWP(DEFAULT_READY_RH_CARTPOS)

        self.trajRetract.addRHOrientWP([0.5004519685854716, -0.8593135617342791, 0.10548947700350815])
        self.trajRetract.addRHOrientWP([0.5004519685854716, -0.8593135617342791, 0.10548947700350815])
        self.trajRetract.addRHOrientWP([0.5004519685854716, -0.8593135617342791, 0.10548947700350815])
        self.trajRetract.addRHOrientWP([0.5004519685854716, -0.8593135617342791, 0.10548947700350815])
        self.trajRetract.addRHOrientWP([0.5004519685854716, -0.8593135617342791, 0.10548947700350815])
        self.trajRetract.addRHOrientWP(DEFAULT_READY_RH_ORIENT)

        self.trajRetract.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        self.trajRetract.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        self.trajRetract.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        self.trajRetract.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        self.trajRetract.addLHCartWP(DEFAULT_READY_LH_CARTPOS)

        self.trajRetract.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        self.trajRetract.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        self.trajRetract.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        self.trajRetract.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        self.trajRetract.addLHOrientWP(DEFAULT_READY_LH_ORIENT)

        self.trajRetract.addPostureWP([0.028212084482967685, 0.028212084482967685,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.19493960990942347, 0.07496994081163051, 0.3741360648243361, 1.398699928093654, 1.4974573555944546, -0.31669671800133464, -0.08953743861272202])
        self.trajRetract.addPostureWP([0.028047416195966585, 0.028047416195966585,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.14405158269023477, 0.08330404179761804, 0.35485388080251956, 1.3998793638319809, 1.5468160099625425, -0.3217818596171365, -0.08263738134922839])
        self.trajRetract.addPostureWP([0.027935113357263137, 0.027935113357263137,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.13328856860260982, 0.07890268167854148, 0.2603418600235295, 1.400536877824891, 1.5541783835179697, -0.3488103334889514, -0.07865545836463488])
        self.trajRetract.addPostureWP([0.027789879631802322, 0.027789879631802322,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.09982449826834842, 0.14084614270850673, -0.0058361512535315884, 1.4018309516255258, 1.556134219337181, -0.34496454788689784, -0.0927880951612021])
        self.trajRetract.addPostureWP([0.028444513491162594, 0.028444513491162594,
            -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
            0.11219453452059695, 0.16546402971632587, -0.11395882910199566, 1.3763986059723414, 1.6190270985068778, -0.37718957675753256, -0.19311388740064422])
        self.trajRetract.addPostureWP(DEFAULT_READY_POSTURE)

        # self.trajRetract.addRHCartWP([0.44428170484806473, -0.2702914994062119, 1.0997219240825615])
        # self.trajRetract.addRHCartWP([0.4203674239044759, -0.2743572663427439, 1.093377210945474])
        # self.trajRetract.addRHCartWP([0.318892510188593, -0.269491700424895, 1.0496129045379383])
        # self.trajRetract.addRHCartWP(DEFAULT_READY_RH_CARTPOS)

        # self.trajRetract.addRHOrientWP([0.019676850165412, -0.9919085050057322, -0.12542064927618982])
        # self.trajRetract.addRHOrientWP([0.019676850165412, -0.9919085050057322, -0.12542064927618982])
        # self.trajRetract.addRHOrientWP([0.019676850165412, -0.9919085050057322, -0.12542064927618982])
        # self.trajRetract.addRHOrientWP(DEFAULT_READY_RH_ORIENT)

        # self.trajRetract.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        # self.trajRetract.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        # self.trajRetract.addLHCartWP(DEFAULT_READY_LH_CARTPOS)
        # self.trajRetract.addLHCartWP(DEFAULT_READY_LH_CARTPOS)

        # self.trajRetract.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        # self.trajRetract.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        # self.trajRetract.addLHOrientWP(DEFAULT_READY_LH_ORIENT)
        # self.trajRetract.addLHOrientWP(DEFAULT_READY_LH_ORIENT)

        # self.trajRetract.addPostureWP([0.06796522908004803, 0.06796522908004803,
        #     -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
        #     0.6089666791320509, 0.08793921769116066, 0.07082243411581694, 1.0425401499369535, 1.641068191538357, -0.20250645610308562, -0.10309210189561308])
        # self.trajRetract.addPostureWP([0.06796522908004803, 0.06796522908004803,
        #     -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
        #     0.5125541454745224, 0.08615348913651269, 0.09537100473609225, 1.1596340637330818, 1.6376422708010987, -0.21181495926711674, -0.12099974279280543])
        # self.trajRetract.addPostureWP([0.06796522908004803, 0.06796522908004803,
        #     -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
        #     0.1295249978515601, 0.08931744864236359, 0.09266642932477151, 1.471017685835546, 1.623110228955735, -0.23047980576603305, -0.04256081682008954])
        # self.trajRetract.addPostureWP(DEFAULT_READY_POSTURE)

    def execute(self, userdata):
        rospy.loginfo('Executing TrajectoryState')

        if self.dreamerInterface.followTrajectory(self.trajExtend):
            rospy.sleep(2.0) # pause 2 seconds
            if self.dreamerInterface.followTrajectory(self.trajShake):
                 rospy.sleep(2.0) # pause 2 seconds
                 if self.dreamerInterface.followTrajectory(self.trajRetract):
                     return "done"
                 else:
                    return "exit"
            else:
                return "exit"
        else:
            return "exit"

class GoBackToReadyState(smach.State):
    """
    A SMACH state that makes the end effectors go back to the ready state,
    which is also the beginning of the GoToIdle trajectory.
    """

    def __init__(self, dreamerInterface, goToIdleTraj):
        """
        The constructor.

        Keyword Parameters:
          - dreamerInterface: The object to which to provide the trajectory.
          - goToIdleTraj: The goToIdle trajectory. This is used to determine the final
                          waypoint of the GoBackToReady trajectory.
        """

        smach.State.__init__(self, outcomes=["done", "exit"])
        self.dreamerInterface = dreamerInterface
        self.goToIdleTraj = goToIdleTraj

    def dist(self, point1, point2):
        return math.sqrt(math.pow(point1[0] - point2[0], 2) + \
                         math.pow(point1[1] - point2[1], 2) + \
                         math.pow(point1[2] - point2[2], 2))

    def execute(self, userdata):
        rospy.loginfo("GoBackToReadyState: Executing GoBackToReadyState")

        # Let's make the trajectory's duration a function of the Cartesian distance to traverse
        rhCurrCartPos = self.dreamerInterface.rightHandCartesianGoalMsg.data
        lhCurrCartPos = self.dreamerInterface.leftHandCartesianGoalMsg.data

        rhFinalCartPos = self.goToIdleTraj.rhCartWP[0]
        lhFinalCartPos = self.goToIdleTraj.lhCartWP[0]

        travelDist = max(self.dist(rhCurrCartPos, rhFinalCartPos), self.dist(lhCurrCartPos, lhFinalCartPos))

        if travelDist < 0.01:  # less than 1cm of movement, return done
            rospy.loginfo("GoBackToReadyState: zero travel distance, returning done")
            return "done"

        travelTime = travelDist / GO_BACK_TO_READY_SPEED
        rospy.loginfo("GoBackToReadyState: distance to travel is {0}, travel time is {1}".format(travelDist, travelTime))

        # Create a trajectory to go back to the start / idle position
        traj = Trajectory.Trajectory("GoBackToReady", travelTime)  # TODO: make the time be proportional to the distance that needs to be traveled

        # Initial goal is current goal
        traj.setInitRHCartWP(rhCurrCartPos)
        traj.setInitLHCartWP(lhCurrCartPos)
        traj.setInitRHOrientWP(self.dreamerInterface.rightHandOrientationGoalMsg.data)
        traj.setInitLHOrientWP(self.dreamerInterface.leftHandOrientationGoalMsg.data)
        traj.setInitPostureWP(self.dreamerInterface.postureGoalMsg.data)

        # repeat the same point twice (this is to ensure the trajectory has sufficient number of points to perform cubic spline)
        traj.addRHCartWP(self.dreamerInterface.rightHandCartesianGoalMsg.data)
        traj.addRHOrientWP(self.dreamerInterface.rightHandOrientationGoalMsg.data)
        traj.addLHCartWP(self.dreamerInterface.leftHandCartesianGoalMsg.data)
        traj.addLHOrientWP(self.dreamerInterface.leftHandOrientationGoalMsg.data)
        traj.addPostureWP(self.dreamerInterface.postureGoalMsg.data)

        traj.addRHCartWP(self.dreamerInterface.rightHandCartesianGoalMsg.data)
        traj.addRHOrientWP(self.dreamerInterface.rightHandOrientationGoalMsg.data)
        traj.addLHCartWP(self.dreamerInterface.leftHandCartesianGoalMsg.data)
        traj.addLHOrientWP(self.dreamerInterface.leftHandOrientationGoalMsg.data)
        traj.addPostureWP(self.dreamerInterface.postureGoalMsg.data)

        # final way point is initial waypoint of GoToIdle trajectory
        traj.addRHCartWP(rhFinalCartPos)
        traj.addRHOrientWP(self.goToIdleTraj.rhOrientWP[0])
        traj.addLHCartWP(lhFinalCartPos)
        traj.addLHOrientWP(self.goToIdleTraj.lhOrientWP[0])
        traj.addPostureWP(self.goToIdleTraj.jPosWP[0])

        # repeat the same final point (this is to ensure the trajectory has sufficient number of points to perform cubic spline)
        traj.addRHCartWP(self.goToIdleTraj.rhCartWP[0])
        traj.addRHOrientWP(self.goToIdleTraj.rhOrientWP[0])
        traj.addLHCartWP(self.goToIdleTraj.lhCartWP[0])
        traj.addLHOrientWP(self.goToIdleTraj.lhOrientWP[0])
        traj.addPostureWP(self.goToIdleTraj.jPosWP[0])

        if self.dreamerInterface.followTrajectory(traj):
            return "done"
        else:
            return "exit"


class AwaitCommandState(smach.State):
    """
    A SMACH state that waits for a command to arrive. It subscribes to a
    ROS topic over which commands are published and triggers a transition
    based on the received command.
    """

    def __init__(self, moveCartesianState, moveOrientationState, goToIdleState):
        """
        The constructor.

        Keyword parameters:
          - moveCartesianState: The SMACH state that moves the cartesian position of the end effector
          - moveOrientationState: The SMACH state that moves the orientation of the end effector
          - goToIdleState: The SMACH state that moves the robot back to the idle state
        """

        # Initialize parent class
        smach.State.__init__(self, outcomes=[
            "go_to_ready",
            "go_back_to_ready",
            "move_position",
            "move_orientation",
            "grasp_end_effector",
            "execute_hand_shake",
            # "execute_wave",
            # "execute_hookem_horns",
            # "execute_demo",
            "done",
            "exit"],
            output_keys=['endEffectorSide', # i.e., "left" or "right"
                         'endEffectorCmd'])  # i.e., "open" or "close"

        self.moveCartesianState = moveCartesianState
        self.moveOrientationState = moveOrientationState
        self.goToIdleState = goToIdleState

        # Initialize local variables
        self.rcvdCmd = False
        self.sleepPeriod = 0.5  # in seconds
        self.cmd = Command.CMD_GOTO_IDLE
        self.isIdle = True  # Initially we are in idle state

        # Register a ROS topic listener
        self.demoCmdSubscriber  = rospy.Subscriber("/demo9/cmd", Int32, self.demo9CmdCallback)
        self.demoDonePublisher = rospy.Publisher("/demo9/done",  Int32, queue_size=1)

        # self.demoCmdSubscriber  = rospy.Subscriber("/demo8/cmd", Int32, self.demo8CmdCallback)
        # self.demoDonePublisher = rospy.Publisher("/demo8/done",  Int32, queue_size=1)

    def demo9CmdCallback(self, msg):
        """
        The callback method of the command ROS topic subscriber.
        """

        self.cmd = msg.data
        self.rcvdCmd = True

    # def demo8CmdCallback(self, msg):
    #     """
    #     The callback method of the command ROS topic subscriber.
    #     """

    #     self.cmd = Command.CMD_EXECUTE_DEMO

    #     # TODO: Get rid of hard-coded demo IDs. CARL Bridge should send this info to us directly.
    #     DEMO_WAVE = 0
    #     DEMO_SHAKE = 1
    #     DEMO_HOOKEM_HORNS = 2

    #     if msg.data == DEMO_WAVE:
    #         self.demoName = "HandWave"
    #     elif msg.data == DEMO_SHAKE:
    #         self.demoName = "HandShake"
    #     elif msg.data == DEMO_HOOKEM_HORNS:
    #         self.demoName = "HookemHorns"
    #     else:
    #         rospy.logwarn("Unknown demo {0}".format(msg.data))
    #         return

    #     self.rcvdCmd = True

    def process1DigitCmd(self, cmd):
        if cmd == Command.CMD_GOTO_READY:

            # Only go to ready if we're in idle    TODO: eventually allow robot to return to ready from anywhere
            if self.isIdle:
                self.isIdle = False
                return "go_to_ready"
            else:
                return "done"

        elif cmd == Command.CMD_GOTO_IDLE:

            # Only go to idle if we're not in idle
            if not self.isIdle:
                self.isIdle = True
                return "go_back_to_ready"
            else:
                return "done"

        elif cmd == Command.CMD_BEHAVIOR_SHAKE:
            return "execute_hand_shake"

        elif cmd == Command.CMD_BEHAVIOR_WAVE:
            return "execute_wave"

        elif cmd == Command.CMD_BEHAVIOR_HOOKEM:
            return "execute_horns"

        elif cmd == Command.CMD_BEHAVIOR_MORE:
            return "execute_demo"

    def process2DigitCmd(self, cmd):
        digit1 = int(cmd / 10)
        digit2 = int(cmd - digit1 * 10)

        print "Two digit command:\n"\
              "  - digit1 = {0}\n"\
              "  - digit2 = {1}".format(digit1, digit2)

        if digit1 == CMD_LEFT_GRIPPER:
            userdata.endEffectorSide = "left"
            if digit2 == CMD_OPEN:
                userData.endEffectorCmd = "open"
            else:
                userData.endEffectorCmd = "close"
        elif digit1 == CMD_RIGHT_HAND:
            userdata.endEffectorSide = "right"
            if digit2 == CMD_OPEN:
                userData.endEffectorCmd = "open"
            else:
                userData.endEffectorCmd = "close"
        else:
            rospy.logerr("Invalid digit 1 of command {0}.".format(cmd))
            return "done"
        return "grasp_end_effector"

    def process3DigitCmd(self, cmd):
        digit1 = int(cmd / 100)
        digit2 = int((cmd - digit1 * 100) / 10)
        digit3 = int(cmd - digit1 * 100 - digit2 * 10)

        print "Three digit command:\n"\
              "  - digit1 = {0}\n"\
              "  - digit2 = {1}\n"\
              "  - digit3 = {2}".format(digit1, digit2, digit3)

        if digit2 == Command.CMD_LEFT_GRIPPER:
            endEffector = "left"
        elif digit2 == Command.CMD_RIGHT_HAND:
            endEffector = "right"
        else:
            rospy.logerr("Invalid second digit of command {0}.".format(cmd))
            return "done"

        if digit3 == Command.CMD_MOVE_LEFT:
            direction = CartesianDirection.LEFT
        elif digit3 == Command.CMD_MOVE_RIGHT:
            direction = CartesianDirection.RIGHT
        elif digit3 == Command.CMD_MOVE_UP:
            direction = CartesianDirection.UP
        elif digit3 == Command.CMD_MOVE_DOWN:
            direction = CartesianDirection.DOWN
        elif digit3 == Command.CMD_MOVE_FORWARD:
            direction = CartesianDirection.FORWARD
        elif digit3 == Command.CMD_MOVE_BACKWARD:
            direction = CartesianDirection.BACKWARD
        else:
            rospy.logerr("Invalid third digit of command {0}.".format(cmd))
            return "done"

        if digit1 == Command.CMD_TRANSLATE:
            self.moveCartesianState.setParameters(endEffector = endEffector, direction = direction)
            return "move_position"
        elif digit1 == Command.CMD_ROTATE:
            self.moveOrientationState.setParameters(endEffector = endEffector, direction = direction)
            return "move_orientation"
        else:
            rospy.logerr("Invalid first digit of command {0}.".format(cmd))
            return "done"

    def execute(self, userdata):
        """
        Waits for a command to arrive. Then processes the command.
        """

        rospy.loginfo('AwaitCommandState: Executing...')

        # Wait for a command to be received. Ingore all commands sent prior to now.

        self.rcvdCmd = False

        while not self.rcvdCmd and not rospy.is_shutdown():
            rospy.sleep(self.sleepPeriod)

        # Parse the command received
        if rospy.is_shutdown():
            return "exit"
        else:
            # Process single digit commands
            if self.cmd < 10:
                return self.process1DigitCmd(self.cmd)

            # Process 2 digit commands
            elif self.cmd > 9 and self.cmd < 100:
                return self.process2DigitCmd(self.cmd)

            # Process 3 digit commands
            elif self.cmd > 99 and self.cmd < 1000:
                return self.process3DigitCmd(self.cmd)

class MoveCartesianState(smach.State):
    """
    A SMACH state that moves the Cartesian position of a point on the robot.
    """

    def __init__(self, dreamerInterface):
        """
        The constructor.

        Keyword Parameters:
          - dreamerInterface: The object providing access to Dreamer hardware.
        """

        smach.State.__init__(self, outcomes=["done", "exit"])
        self.dreamerInterface = dreamerInterface
        self.trajGen = TrapezoidVelocityTrajGen.TrapezoidVelocityTrajGen()

    def setParameters(self, endEffector, direction):
        """
        Sets the end effector and movement direction parameters.

        Keyword Parameters:
          - dreamerInterface: The object providing access to Dreamer hardware.
          - endEffector: Which end effector to adjust.
          - direction: The direction to adjust the end effector's Cartesian position.
        """

        self.endEffector = endEffector
        self.direction = direction

    def directionToString(self, direction):
        """
        Returns a string representation of the direction command.

        Keyword Parameters:
          - direction: the direction to convert into a string.
        """

        if direction == CartesianDirection.UP:
            return "UP"
        elif direction == CartesianDirection.DOWN:
            return "DOWN"
        elif direction == CartesianDirection.LEFT:
            return "LEFT"
        elif direction == CartesianDirection.RIGHT:
            return "RIGHT"
        elif direction == CartesianDirection.FORWARD:
            return "FORWARD"
        elif direction == CartesianDirection.BACKWARD:
            return "BACKWARD"
        else:
            return "UNKNOWN"

    def axisNameToString(self, axisID):
        if axisID == 0:
            return "X"
        if axisID == 1:
            return "Y"
        if axisID == 2:
            return "Z"

    def execute(self, userdata):
        rospy.loginfo('MoveCartesianState: Executing, end effector = {0}, direction = {1}'.format(
            self.endEffector, self.directionToString(self.direction)))

        # Determine the current Cartesian position
        if self.endEffector == "right":
            self.origCartesianPosition = self.dreamerInterface.rightHandCartesianGoalMsg.data
        else:
            self.origCartesianPosition = self.dreamerInterface.leftHandCartesianGoalMsg.data

        if self.origCartesianPosition == None:
            rospy.loginfo("MoveCartesianState: ERROR: Unable to get current Cartesian position. Aborting the move. Returning done.")
            return "done"

        # Compute new Cartesian position

        if self.direction == CartesianDirection.FORWARD:
            oldPos = self.origCartesianPosition[X_AXIS]
            newPos = self.origCartesianPosition[X_AXIS] + CARTESIAN_MOVE_DELTA
            self.axisOfMovement = X_AXIS

        if self.direction == CartesianDirection.BACKWARD:
            oldPos = self.origCartesianPosition[X_AXIS]
            newPos = self.origCartesianPosition[X_AXIS] - CARTESIAN_MOVE_DELTA
            self.axisOfMovement = X_AXIS

        if self.direction == CartesianDirection.LEFT:
            oldPos = self.origCartesianPosition[Y_AXIS]
            newPos = self.origCartesianPosition[Y_AXIS] + CARTESIAN_MOVE_DELTA
            self.axisOfMovement = Y_AXIS

        if self.direction == CartesianDirection.RIGHT:
            oldPos = self.origCartesianPosition[Y_AXIS]
            newPos = self.origCartesianPosition[Y_AXIS] - CARTESIAN_MOVE_DELTA
            self.axisOfMovement = Y_AXIS

        if self.direction == CartesianDirection.UP:
            oldPos = self.origCartesianPosition[Z_AXIS]
            newPos = self.origCartesianPosition[Z_AXIS] + CARTESIAN_MOVE_DELTA
            self.axisOfMovement = Z_AXIS

        if self.direction == CartesianDirection.DOWN:
            oldPos = self.origCartesianPosition[Z_AXIS]
            newPos = self.origCartesianPosition[Z_AXIS] - CARTESIAN_MOVE_DELTA
            self.axisOfMovement = Z_AXIS

        rospy.loginfo("MoveCartesianState: Modifying goal Cartesian position of {0} axis to be from {1} to {2}".format(
            self.axisNameToString(self.axisOfMovement), oldPos, newPos))

        # Initialize the trajectory generator
        self.trajGen.init(oldPos, newPos, TRAVEL_SPEED, ACCELERATION, DECELERATION,
            TRAJECTORY_UPDATE_FREQUENCY)

        # Start the trajectory! The callback method is updateTrajGoals(...), which is defined below.
        self.trajGen.start(self)

        # Wait 2 seconds to allow convergence before returning
        # rospy.sleep(2)

        if rospy.is_shutdown():
             return "exit"
        else:
            return "done"

    def updateTrajGoals(self, goalPos, goalVel, done):

        # Initialize the new Cartesian position to be the original Cartesian position
        newCartesianPosition = self.origCartesianPosition

        # Save the new goal Cartesian position
        if self.axisOfMovement == X_AXIS:
            newCartesianPosition[X_AXIS] = goalPos
        elif self.axisOfMovement == Y_AXIS:
            newCartesianPosition[Y_AXIS] = goalPos
        else:
            newCartesianPosition[Z_AXIS] = goalPos

        # rospy.loginfo("MoveCartesianState: Updating goal of {0} and to be {1}.".format(
        #     self.endEffector, newCartesianPosition))

        if self.endEffector == "right":
            self.dreamerInterface.updateRightHandPosition(newCartesianPosition)
        else:
            self.dreamerInterface.updateLeftHandPosition(newCartesianPosition)

class MoveOrientationState(smach.State):
    """
    A SMACH state that moves the orientation of the robot's end effector.
    """

    def __init__(self, dreamerInterface):
        """
        The constructor.

        Keyword Parameters:
          - dreamerInterface: The object providing access to Dreamer hardware.
        """

        smach.State.__init__(self, outcomes=["done", "exit"])
        self.dreamerInterface = dreamerInterface
        self.trajGen = TrapezoidVelocityTrajGen.TrapezoidVelocityTrajGen()

    def setParameters(self, endEffector, direction):
        """
        Sets the end effector and movement direction parameters.

        Keyword Parameters:
          - dreamerInterface: The object providing access to Dreamer hardware.
          - endEffector: Which end effector to adjust.
          - direction: The direction to adjust the end effector's Cartesian position.
        """

        self.endEffector = endEffector
        self.direction = direction

    def directionToString(self, direction):
        """
        Returns a string representation of the direction command.

        Keyword Parameters:
          - direction: the direction to convert into a string.
        """

        if direction == CartesianDirection.UP:
            return "Pitch UP"
        elif direction == CartesianDirection.DOWN:
            return "Pitch DOWN"
        elif direction == CartesianDirection.LEFT:
            return "Roll LEFT"
        elif direction == CartesianDirection.RIGHT:
            return "Roll RIGHT"
        elif direction == CartesianDirection.FORWARD:
            return "Yaw Left"
        elif direction == CartesianDirection.BACKWARD:
            return "Yaw Right"
        else:
            return "UNKNOWN"

    # def axisNameToString(self, axisID):
    #     if axisID == 0:
    #         return "X"
    #     if axisID == 1:
    #         return "Y"
    #     if axisID == 2:
    #         return "Z"

    def execute(self, userdata):
        rospy.loginfo('MoveOrientationState: Executing, end effector = {0}, direction = {1}'.format(
            self.endEffector, self.directionToString(self.direction)))

        # Determine the current orientation
        if self.endEffector == "right":
            self.origOrientation = self.dreamerInterface.rightHandOrientationGoalMsg.data
        else:
            self.origOrientation = self.dreamerInterface.leftHandOrientationGoalMsg.data

        if self.origOrientation == None:
            rospy.loginfo("MoveOrientationState: ERROR: Unable to get current orientation. "\
                          "Aborting the move. Returning done.")
            return "done"

        # Compute new orientation

        rospy.loginfo("TODO: Implement orientation movements.")

        # if self.direction == CartesianDirection.FORWARD:
        #     oldPos = self.origOrientation[X_AXIS]
        #     newPos = self.origOrientation[X_AXIS] + CARTESIAN_MOVE_DELTA
        #     self.axisOfMovement = X_AXIS

        # if self.direction == CartesianDirection.BACKWARD:
        #     oldPos = self.origOrientation[X_AXIS]
        #     newPos = self.origOrientation[X_AXIS] - CARTESIAN_MOVE_DELTA
        #     self.axisOfMovement = X_AXIS

        # if self.direction == CartesianDirection.LEFT:
        #     oldPos = self.origOrientation[Y_AXIS]
        #     newPos = self.origOrientation[Y_AXIS] + CARTESIAN_MOVE_DELTA
        #     self.axisOfMovement = Y_AXIS

        # if self.direction == CartesianDirection.RIGHT:
        #     oldPos = self.origOrientation[Y_AXIS]
        #     newPos = self.origOrientation[Y_AXIS] - CARTESIAN_MOVE_DELTA
        #     self.axisOfMovement = Y_AXIS

        # if self.direction == CartesianDirection.UP:
        #     oldPos = self.origOrientation[Z_AXIS]
        #     newPos = self.origOrientation[Z_AXIS] + CARTESIAN_MOVE_DELTA
        #     self.axisOfMovement = Z_AXIS

        # if self.direction == CartesianDirection.DOWN:
        #     oldPos = self.origOrientation[Z_AXIS]
        #     newPos = self.origOrientation[Z_AXIS] - CARTESIAN_MOVE_DELTA
        #     self.axisOfMovement = Z_AXIS

        # rospy.loginfo("MoveOrientationState: Modifying goal orientation of {0} axis to be from {1} to {2}".format(
        #     self.axisNameToString(self.axisOfMovement), oldPos, newPos))

        # # Initialize the trajectory generator
        # self.trajGen.init(oldPos, newPos, TRAVEL_SPEED, ACCELERATION, DECELERATION,
        #     TRAJECTORY_UPDATE_FREQUENCY)

        # # Start the trajectory! The callback method is updateTrajGoals(...), which is defined below.
        # self.trajGen.start(self)

        # Wait 2 seconds to allow convergence before returning
        # rospy.sleep(2)

        if rospy.is_shutdown():
             return "exit"
        else:
            return "done"

    def updateTrajGoals(self, goalPos, goalVel, done):

        # TODO

        # Initialize the new Cartesian position to be the original Cartesian position
        newOrientation = self.origOrientation

        # # Save the new goal Cartesian position
        # if self.axisOfMovement == X_AXIS:
        #     newOrientation[X_AXIS] = goalPos
        # elif self.axisOfMovement == Y_AXIS:
        #     newOrientation[Y_AXIS] = goalPos
        # else:
        #     newOrientation[Z_AXIS] = goalPos

        # rospy.loginfo("MoveCartesianState: Updating goal of {0} and to be {1}.".format(
        #     self.endEffector, newOrientation))

        if self.endEffector == "right":
            self.dreamerInterface.updateRightHandOrientation(newOrientation)
        else:
            self.dreamerInterface.updateLeftHandOrientation(newOrientation)

class EndEffectorState(smach.State):
    """
    A SMACH state that toggles the state of an end effector.
    """

    def __init__(self, dreamerInterface):
        """
        The constructor.

        Keyword Parameters:
          - dreamerInterface: The object to which to provide the trajectory.
        """

        smach.State.__init__(self, outcomes=["done", "exit"], input_keys=['endEffectorSide'])
        self.dreamerInterface = dreamerInterface
        self.isClosed = False  # Assume initial state is relaxed

    def execute(self, userdata):
        rospy.loginfo('Executing EndEffectorState, side = {0}'.format(userdata.endEffectorSide))

        if userdata.endEffectorSide == "right":
            if self.isClosed:
                self.dreamerInterface.openRightHand()
            else:
                self.dreamerInterface.closeRightHand()
        else:
            if self.isClosed:
                self.dreamerInterface.openLeftGripper()
            else:
                self.dreamerInterface.closeLeftGripper()

        self.isClosed = not self.isClosed

        # wait 2 seconds to allow convergence before returning
        rospy.sleep(2)

        if rospy.is_shutdown():
             return "exit"
        else:
            return "done"

class SleepState(smach.State):
    """
    Makes the robot pause for a specified amount of time.
    """

    def __init__(self, goodResult, period):
        smach.State.__init__(self, outcomes=[goodResult, "exit"])
        self.goodResult = goodResult
        self.period = period

    def execute(self, userdata):
        rospy.loginfo("Executing SleepState")
        rospy.sleep(self.period)

        if rospy.is_shutdown():
            return "exit"
        else:
            return self.goodResult

# class ExecuteDemoState(smach.State):
#     """
#     Executes a demo.
#     """

#     def __init__(self, dreamerInterface):
#         smach.State.__init__(self, outcomes=["done", "exit"], input_keys=['demoName'])

#         # Instantiate the previous demos
#         self.handWaveDemo = Demo4_HandWave.Demo4_HandWave(dreamerInterface)
#         self.handShakeDemo = Demo5_HandShake.Demo5_HandShake(dreamerInterface)
#         self.hookemHornsDemo = Demo7_HookemHorns.Demo7_HookemHorns(dreamerInterface)

#     def execute(self, userdata):
#         rospy.loginfo('Executing demo {0}'.format(userdata.demoName))

#         if userdata.demoName == "HandWave":
#             print "Starting the Hand Wave Demo!"
#             self.handWaveDemo.run(enablePrompts = False)
#         elif userdata.demoName == "HandShake":
#             print "Starting the Hand Shake Demo!"
#             self.handShakeDemo.run(enablePrompts = False)
#         elif userdata.demoName == "HookemHorns":
#             print "Starting the Hook'em Horns Demo!"
#             self.hookemHornsDemo.run(enablePrompts = False)
#         else:
#             rospy.logwarn("Unknown demo {0}".format(userdata.demoName))

#         if rospy.is_shutdown():
#              return "exit"
#         else:
#             return "done"

class Demo9_CARL_Telemanipulation:
    """
    The primary class that implement's the demo's FSM.
    """

    def __init__(self):
        self.dreamerInterface = DreamerInterface.DreamerInterface(ENABLE_USER_PROMPTS)

    def createTrajectories(self):

        # ==============================================================================================
        # Define the GoToReady trajectory
        self.trajGoToReady = Trajectory.Trajectory("GoToReady", TIME_GO_TO_READY)

        # These are the initial values as specified in the YAML ControlIt! configuration file
        self.trajGoToReady.setInitRHCartWP([0.033912978219317776, -0.29726881641499886, 0.82])
        self.trajGoToReady.setInitLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoToReady.setInitRHOrientWP([1.0, 0.0, 0.0])
        self.trajGoToReady.setInitLHOrientWP([1.0, 0.0, 0.0])
        self.trajGoToReady.setInitPostureWP(DEFAULT_POSTURE)

        self.trajGoToReady.addRHCartWP([0.019903910090688474, -0.28423307267223147, 0.9179288590591458])
        self.trajGoToReady.addRHCartWP([-0.055152798770261954, -0.2907526623508046, 1.009663652974324])
        self.trajGoToReady.addRHCartWP([-0.03366873622218044, -0.40992725074781894, 1.1144948070701866])
        self.trajGoToReady.addRHCartWP([0.11866831717348489, -0.4101100845056917, 1.209699047600146])
        self.trajGoToReady.addRHCartWP([0.21649227857092893, -0.3006839904787592, 1.1140502834793191])
        self.trajGoToReady.addRHCartWP(DEFAULT_READY_RH_CARTPOS)

        self.trajGoToReady.addRHOrientWP([0.8950968852599132, 0.26432788250814326, 0.3590714922223199])
        self.trajGoToReady.addRHOrientWP([0.8944226954968388, 0.33098423072776184, 0.3007615015086225])
        self.trajGoToReady.addRHOrientWP([0.8994250702615956, 0.22626156457297464, 0.3739521993275524])
        self.trajGoToReady.addRHOrientWP([0.19818667912613866, -0.8161433027447201, 0.5428002851895832])
        self.trajGoToReady.addRHOrientWP([0.260956993686226, -0.8736061290033836, 0.4107478287392042])
        self.trajGoToReady.addRHOrientWP(DEFAULT_READY_RH_ORIENT)

        self.trajGoToReady.addLHCartWP([0.019903910090688474, 0.28423307267223147, 0.9179288590591458])
        self.trajGoToReady.addLHCartWP([-0.055152798770261954, 0.2907526623508046, 1.009663652974324])
        self.trajGoToReady.addLHCartWP([-0.03366873622218044, 0.40992725074781894, 1.1144948070701866])
        self.trajGoToReady.addLHCartWP([0.11866831717348489, 0.4101100845056917, 1.209699047600146])
        self.trajGoToReady.addLHCartWP([0.21649227857092893, 0.3006839904787592, 1.1140502834793191])
        self.trajGoToReady.addLHCartWP(DEFAULT_READY_LH_CARTPOS)

        self.trajGoToReady.addLHOrientWP([0.8950968852599132, -0.26432788250814326, 0.3590714922223199])
        self.trajGoToReady.addLHOrientWP([0.8944226954968388, -0.33098423072776184, 0.3007615015086225])
        self.trajGoToReady.addLHOrientWP([0.8994250702615956, -0.22626156457297464, 0.3739521993275524])
        self.trajGoToReady.addLHOrientWP([0.19818667912613866, 0.8161433027447201, 0.5428002851895832])
        self.trajGoToReady.addLHOrientWP([0.260956993686226, 0.8736061290033836, 0.4107478287392042])
        self.trajGoToReady.addLHOrientWP(DEFAULT_READY_LH_ORIENT)


        self.trajGoToReady.addPostureWP([0.06826499288341317, 0.06826499288341317,
            -0.6249282444166423,  0.3079607416653748,  -0.1220981510225299,  1.3675006234559883, 0.06394316468492173, -0.20422693251592328, 0.06223224746326836,
            -0.6249282444166423,  0.3079607416653748,  -0.1220981510225299,  1.3675006234559883, 0.06394316468492173, -0.20422693251592328, 0.06223224746326836])
        self.trajGoToReady.addPostureWP([0.0686363596318602,  0.0686363596318602,
            -1.0914342991625676,  0.39040871074764566, -0.03720209764435387, 1.7583823306095314, 0.05438773164693069, -0.20257591921666193, 0.06386553930484179,
            -1.0914342991625676,  0.39040871074764566, -0.03720209764435387, 1.7583823306095314, 0.05438773164693069, -0.20257591921666193, 0.06386553930484179])
        self.trajGoToReady.addPostureWP([0.06804075180539401, 0.06804075180539401,
            -1.3637873691001094,  0.3926057912988488,  0.575755053425441,    1.9732992187122156, 0.29999797251313004, -0.20309827518257023, 0.05586603055643467,
            -1.3637873691001094,  0.3926057912988488,  0.575755053425441,    1.9732992187122156, 0.29999797251313004, -0.20309827518257023, 0.05586603055643467])
        self.trajGoToReady.addPostureWP([0.06818415549992426, 0.06818415549992426,
            -0.8497599545494692,  0.47079074342878563, 0.8355038507753617,   2.2318590905389852, 1.8475059506175733,  -0.405570582208143,   -0.0277359315904628,
            -0.8497599545494692,  0.47079074342878563, 0.8355038507753617,   2.2318590905389852, 1.8475059506175733,  -0.405570582208143,   -0.0277359315904628])
        self.trajGoToReady.addPostureWP([0.06794500584573498, 0.06794500584573498,
            -0.24608246913199228, 0.13441397755549533, 0.2542869735593113,   2.0227000417984633, 1.3670468713459782,  -0.45978204939890815, 0.030219082955597457,
            -0.24608246913199228, 0.13441397755549533, 0.2542869735593113,   2.0227000417984633, 1.3670468713459782,  -0.45978204939890815, 0.030219082955597457])
        self.trajGoToReady.addPostureWP(DEFAULT_READY_POSTURE)

        # ==============================================================================================
        # Define the GoToIdle trajectory
        self.trajGoToIdle = Trajectory.Trajectory("GoToIdle", TIME_GO_TO_IDLE)
        self.trajGoToIdle.setPrevTraj(self.trajGoToReady)                        # This trajectory always starts where the GoToReady trajectory ends

        # 2015.01.06 Trajectory
        self.trajGoToIdle.addRHCartWP([0.25822435038901964, -0.1895604971725577, 1.0461857180093073])
        self.trajGoToIdle.addRHCartWP([0.21649227857092893, -0.3006839904787592, 1.1140502834793191])
        self.trajGoToIdle.addRHCartWP([0.11866831717348489, -0.4101100845056917, 1.209699047600146])
        self.trajGoToIdle.addRHCartWP([-0.03366873622218044, -0.40992725074781894, 1.1144948070701866])
        self.trajGoToIdle.addRHCartWP([-0.055152798770261954, -0.2907526623508046, 1.009663652974324])
        self.trajGoToIdle.addRHCartWP([0.019903910090688474, -0.28423307267223147, 0.9179288590591458])
        self.trajGoToIdle.addRHCartWP([0.033912978219317776, -0.29726881641499886, 0.82]) # Matches the start of trajectory GoToReady

        self.trajGoToIdle.addRHOrientWP([0.5409881394605172, -0.8191390472602035, 0.19063854336595773])
        self.trajGoToIdle.addRHOrientWP([0.260956993686226, -0.8736061290033836, 0.4107478287392042])
        self.trajGoToIdle.addRHOrientWP([0.19818667912613866, -0.8161433027447201, 0.5428002851895832])
        self.trajGoToIdle.addRHOrientWP([0.8994250702615956, 0.22626156457297464, 0.3739521993275524])
        self.trajGoToIdle.addRHOrientWP([0.8944226954968388, 0.33098423072776184, 0.3007615015086225])
        self.trajGoToIdle.addRHOrientWP([0.8950968852599132, 0.26432788250814326, 0.3590714922223199])
        self.trajGoToIdle.addRHOrientWP([1.0, 0.0, 0.0]) # Matches the start of trajectory GoToReady

        self.trajGoToIdle.addLHCartWP([0.25822435038901964, 0.1895604971725577, 1.0461857180093073])
        self.trajGoToIdle.addLHCartWP([0.21649227857092893, 0.3006839904787592, 1.1140502834793191])
        self.trajGoToIdle.addLHCartWP([0.11866831717348489, 0.4101100845056917, 1.209699047600146])
        self.trajGoToIdle.addLHCartWP([-0.03366873622218044, 0.40992725074781894, 1.1144948070701866])
        self.trajGoToIdle.addLHCartWP([-0.055152798770261954, 0.2907526623508046, 1.009663652974324])
        self.trajGoToIdle.addLHCartWP([0.019903910090688474, 0.28423307267223147, 0.9179288590591458])
        self.trajGoToIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82]) # Matches the start of trajectory GoToReady

        self.trajGoToIdle.addLHOrientWP([0.5409881394605172, 0.8191390472602035, 0.19063854336595773])
        self.trajGoToIdle.addLHOrientWP([0.260956993686226, 0.8736061290033836, 0.4107478287392042])
        self.trajGoToIdle.addLHOrientWP([0.19818667912613866, 0.8161433027447201, 0.5428002851895832])
        self.trajGoToIdle.addLHOrientWP([0.8994250702615956, -0.22626156457297464, 0.3739521993275524])
        self.trajGoToIdle.addLHOrientWP([0.8944226954968388, -0.33098423072776184, 0.3007615015086225])
        self.trajGoToIdle.addLHOrientWP([0.8950968852599132, -0.26432788250814326, 0.3590714922223199])
        self.trajGoToIdle.addLHOrientWP([1.0, 0.0, 0.0]) # Matches the start of trajectory GoToReady

        self.trajGoToIdle.addPostureWP([0.06796522908004803, 0.06796522908004803,                                                  # torso
                       -0.08569654146540764, 0.07021124925432169,                    0, 1.7194162945362514, 1.51, -0.07, -0.18,    # left arm
                       -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])   # right arm
        self.trajGoToIdle.addPostureWP([0.06794500584573498, 0.06794500584573498, -0.24608246913199228, 0.13441397755549533, 0.2542869735593113,   2.0227000417984633, 1.3670468713459782,  -0.45978204939890815, 0.030219082955597457, -0.24608246913199228, 0.13441397755549533, 0.2542869735593113,   2.0227000417984633, 1.3670468713459782,  -0.45978204939890815, 0.030219082955597457])
        self.trajGoToIdle.addPostureWP([0.06818415549992426, 0.06818415549992426, -0.8497599545494692,  0.47079074342878563, 0.8355038507753617,   2.2318590905389852, 1.8475059506175733,  -0.405570582208143,   -0.0277359315904628, -0.8497599545494692,  0.47079074342878563, 0.8355038507753617,   2.2318590905389852, 1.8475059506175733,  -0.405570582208143,   -0.0277359315904628])
        self.trajGoToIdle.addPostureWP([0.06804075180539401, 0.06804075180539401, -1.3637873691001094,  0.3926057912988488,  0.575755053425441,    1.9732992187122156, 0.29999797251313004, -0.20309827518257023, 0.05586603055643467, -1.3637873691001094,  0.3926057912988488,  0.575755053425441,    1.9732992187122156, 0.29999797251313004, -0.20309827518257023, 0.05586603055643467])
        self.trajGoToIdle.addPostureWP([0.0686363596318602,  0.0686363596318602,  -1.0914342991625676,  0.39040871074764566, -0.03720209764435387, 1.7583823306095314, 0.05438773164693069, -0.20257591921666193, 0.06386553930484179, -1.0914342991625676,  0.39040871074764566, -0.03720209764435387, 1.7583823306095314, 0.05438773164693069, -0.20257591921666193, 0.06386553930484179])
        self.trajGoToIdle.addPostureWP([0.06826499288341317, 0.06826499288341317, -0.6249282444166423,  0.3079607416653748,  -0.1220981510225299,  1.3675006234559883, 0.06394316468492173, -0.20422693251592328, 0.06223224746326836, -0.6249282444166423,  0.3079607416653748,  -0.1220981510225299,  1.3675006234559883, 0.06394316468492173, -0.20422693251592328, 0.06223224746326836])
        self.trajGoToIdle.addPostureWP(DEFAULT_POSTURE) # Matches the start of trajectory GoToReady

    def createFSM(self):
        # define the states
        moveCartesianState = MoveCartesianState(dreamerInterface = self.dreamerInterface)
        moveOrientationState = MoveOrientationState(dreamerInterface = self.dreamerInterface)

        goToReadyState = TrajectoryState(self.dreamerInterface, self.trajGoToReady)
        goToIdleState = TrajectoryState(self.dreamerInterface, self.trajGoToIdle)
        goBackToReadyState = GoBackToReadyState(self.dreamerInterface, self.trajGoToIdle)
        awaitCommandState = AwaitCommandState(
            moveCartesianState = moveCartesianState,
            moveOrientationState = moveOrientationState,
            goToIdleState = goToIdleState)
        # executeDemoState = ExecuteDemoState(self.dreamerInterface)
        endEffectorState = EndEffectorState(self.dreamerInterface)

        shakeHandState = TrajectoryShakeHands(self.dreamerInterface, self.trajGoToReady)

        # wire the states into a FSM
        self.fsm = smach.StateMachine(outcomes=['exit'])
        self.fsm.userdata.endEffectorSide = "right"
        self.fsm.userdata.demoName = "none"

        with self.fsm:

            smach.StateMachine.add("AwaitCommandState", awaitCommandState,
                transitions={"go_to_ready":"GoToReadyState",
                             "go_back_to_ready":"GoBackToReadyState",
                             "move_position":"MoveCartesianState",
                             "move_orientation":"MoveOrientationState",
                             "grasp_end_effector":"EndEffectorState",
                             # "execute_demo":"ExecuteDemoState",
                             "execute_hand_shake":"ShakeHandState",
                             # "execute_wave":"WaveState",
                             # "execute_hookem_horns":"HornsState",
                             "done":"AwaitCommandState",
                             "exit":"exit"},
                remapping={'endEffectorSide':'endEffectorSide'})

            smach.StateMachine.add("GoToReadyState", goToReadyState,
                transitions={'done':'AwaitCommandState',
                             'exit':'exit'})

            smach.StateMachine.add("GoToIdleState", goToIdleState,
                transitions={'done':'AwaitCommandState',
                             'exit':'exit'})

            smach.StateMachine.add("GoBackToReadyState", goBackToReadyState,
                transitions={'done':'GoToIdleState',
                             'exit':'exit'})

            smach.StateMachine.add("MoveCartesianState", moveCartesianState,
                transitions={'done':'AwaitCommandState',
                             'exit':'exit'})

            smach.StateMachine.add("MoveOrientationState", moveOrientationState,
                transitions={'done':'AwaitCommandState',
                             'exit':'exit'})

            smach.StateMachine.add("EndEffectorState", endEffectorState,
                transitions={'done':'AwaitCommandState',
                             'exit':'exit'},
                remapping={'endEffectorSide':'endEffectorSide',
                           'endEffectorCmd':'endEffectorCmd'})

            # smach.StateMachine.add("ExecuteDemoState", executeDemoState,
            #     transitions={'done':'AwaitCommandState',
            #                  'exit':'exit'},
            #     remapping={'demoName':'demoName'})

            smach.StateMachine.add("ShakeHandState", shakeHandState,
                transitions={'done':'AwaitCommandState',
                             'exit':'exit'})

    def run(self):
        """
        Runs the Cartesian and orientation demo 9 behavior.
        """

        if not self.dreamerInterface.connectToControlIt(DEFAULT_POSTURE):
            return

        self.createTrajectories()
        self.createFSM()

        # Create and start the introspection server
        sis = smach_ros.IntrospectionServer('server_name', self.fsm, '/SM_ROOT')
        sis.start()

        if ENABLE_USER_PROMPTS:
            index = raw_input("Start demo? Y/n\n")
            if index == "N" or index == "n":
                return

        outcome = self.fsm.execute()

        print "Demo 9 done, waiting until ctrl+c is hit..."
        rospy.spin()  # just to prevent this node from exiting
        sis.stop()


# Main method
if __name__ == "__main__":
    rospy.init_node('Demo9_CARL_Telemanipulation', anonymous=True)
    demo = Demo9_CARL_Telemanipulation()
    demo.run()

    print "Demo 9 done, waiting until ctrl+c is hit..."
    rospy.spin()  # just to prevent this node from exiting
