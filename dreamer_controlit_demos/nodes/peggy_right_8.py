#!/usr/bin/env python

'''
Real-Time Experiments:
This uses both posture and orientation control.
'''

import sys, getopt     # for getting and parsing command line arguments
import time
import math
import threading
import rospy
import smach
import smach_ros

from std_msgs.msg import Float64, Float64MultiArray, MultiArrayDimension, Bool, Int32

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import DreamerInterface
import Trajectory
import TrajectoryGeneratorCubicSpline

ENABLE_USER_PROMPTS = False

# Shoulder abductors about 10 degrees away from body and elbows bent 90 degrees
# DEFAULT_POSTURE = [0.0, 0.0,                                    # torso
#                    0.0, 0.174532925, 0.0, 1.57, 0.0, 0.0, 0.0,  # left arm
#                    0.0, 0.174532925, 0.0, 1.57, 0.0, 0.0, 0.0]  # right arm

# Shoulder abductors and elbows at about 10 degrees
DEFAULT_POSTURE = [0.0, 0.0,                                    # torso
                   0.0, 0.174532925, 0.0, 0.174532925, 0.0, 0.0, 0.0,  # left arm
                   0.0, 0.174532925, 0.0, 0.174532925, 0.0, 0.0, 0.0]  # right arm

# The time each trajectory should take
TIME_GO_TO_GRASP = 5.0
TIME_TEST_TRAJECTORY = 2.0
TIME_GO_TO_LOCATION = 3.0
TIME_DROP = 2.0
TIME_GO_TO_IDLE = 4.0


class TrajectoryState(smach.State):
    def __init__(self, dreamerInterface, traj):
        smach.State.__init__(self, outcomes=["done", "exit"])
        self.dreamerInterface = dreamerInterface
        self.traj = traj

    def execute(self, userdata):
        rospy.loginfo('Executing TrajectoryState')

        print  "FIX ME..."

        if self.dreamerInterface.followTrajectory(self.traj):
            return "done"
        else:
            return "exit"

class EnablePowerGraspState(smach.State):
    def __init__(self, dreamerInterface):
        smach.State.__init__(self, outcomes=["done", "exit"])
        self.dreamerInterface = dreamerInterface

    def execute(self, userdata):
        rospy.loginfo('Executing EnablePowerGraspState')

        self.dreamerInterface.rightHandCmdMsg.data = True
        self.dreamerInterface.rightHandCmdPublisher.publish(self.dreamerInterface.rightHandCmdMsg)

        rospy.sleep(5) # allow fingers to move

        if rospy.is_shutdown():
            return "exit"
        else:
            return "done"

class DisablePowerGraspState(smach.State):
    def __init__(self, dreamerInterface):
        smach.State.__init__(self, outcomes=["done", "exit"])
        self.dreamerInterface = dreamerInterface

    def execute(self, userdata):
        rospy.loginfo('Executing DisablePowerGraspState')

        self.dreamerInterface.rightHandCmdMsg.data = False
        self.dreamerInterface.rightHandCmdPublisher.publish(self.dreamerInterface.rightHandCmdMsg)

        rospy.sleep(5) # allow fingers to move

        if rospy.is_shutdown():
            return "exit"
        else:
            return "done"

class SleepState(smach.State):
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

class TestTrajectory:
    def __init__(self):
        self.dreamerInterface = DreamerInterface.DreamerInterface(enableUserPrompts = ENABLE_USER_PROMPTS, useQuaternionControl = True)

    def createTrajectories(self):

	# get object position and orientation from sys.argv
	if len(sys.argv) < 8:
            print 'Input argument is not enough! Abort!'
            sys.exit(1)
        obj_pos = map(float, sys.argv[1:4])
        obj_ori = map(float, sys.argv[4:])

        # ==============================================================================================
        #Define the GoToGrasp trajectory
        self.trajGoGrasp = Trajectory.Trajectory("GoToReady", TIME_GO_TO_GRASP)

        # These are the initial values as specified in the YAML ControlIt! configuration file
        self.trajGoGrasp.setInitRHCartWP([0.033912978219317776, -0.29726881641499886, 0.82])
        self.trajGoGrasp.setInitRHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.setInitLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.setInitLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.setInitPostureWP(DEFAULT_POSTURE)

        # right arm Position	
        self.trajGoGrasp.addRHCartWP([-0.050567047483583594, -0.21042238743220334, 0.78318269732732])
        self.trajGoGrasp.addRHCartWP([-0.04383549622470033, -0.32895888103292337, 0.801589241376078])
        self.trajGoGrasp.addRHCartWP([-0.012090112071880604, -0.45132667825620776, 0.851573020706654])
        self.trajGoGrasp.addRHCartWP([0.03227073981506992, -0.504714369834375, 0.892026287956605])
        self.trajGoGrasp.addRHCartWP([0.053785523415727995, -0.42529648734639197, 0.860078355789907])
        self.trajGoGrasp.addRHCartWP([0.10199973392700928, -0.3645362396646155, 0.898570586114738])
        self.trajGoGrasp.addRHCartWP([0.18006173116275434, -0.29995746764424497, 0.984131289844163])
        self.trajGoGrasp.addRHCartWP([0.26492500399802366, -0.20180132935756945, 0.995510359317888])
        #self.trajGoGrasp.addRHCartWP([0.28728696217490696, -0.14360147287818184, 1.025967881044474])
        self.trajGoGrasp.addRHCartWP(obj_pos)


        # right arm Orientation	
        self.trajGoGrasp.addRHOrientWP([-0.19084049835399294, 0.9801326491244801, 0.05369551983802947, 0.00605685211576487])
        self.trajGoGrasp.addRHOrientWP([-0.0846462672654739, 0.9943312924028822, 0.06420078584604186, 0.00430691109574287])
        self.trajGoGrasp.addRHOrientWP([0.0034072436652181153, 0.986275611103743, -0.11052219700995891, 0.122611800413483])
        self.trajGoGrasp.addRHOrientWP([-0.1588384915726129, 0.81468009880237, 0.1567662060191347, 0.535248565491458])
        self.trajGoGrasp.addRHOrientWP([-0.23917696425078697, 0.7599756088137017, 0.1803355798888413, 0.576810655593476])
        self.trajGoGrasp.addRHOrientWP([-0.26261069233194007, 0.6014822890021102, 0.1715628969429657, 0.734725018413187])
        self.trajGoGrasp.addRHOrientWP([-0.28161466546179365, 0.43825748924303226, 0.2887607439993108, 0.803268813065837])
        self.trajGoGrasp.addRHOrientWP([-0.32715192074437427, 0.49186697302287863, 0.1902052202806057, 0.784130394628714])
        #self.trajTest.addRHOrientWP([-0.3242288649080795, 0.6847128077489325, 0.04957358009436334, 0.65083521280092])
        self.trajGoGrasp.addRHOrientWP(obj_ori)


        # left arm Position does not move
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        #self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        #self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])


        # left arm Orientation does not move
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        #self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        #self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])


        # trajectory
        self.trajGoGrasp.addPostureWP([-0.0664330463486,-0.0664330463486, # torso down
             -0.278467939604,0.0860061567895,-0.110560003988,0.539105155681,0.155230657624,-0.368227235264,-0.190612562152, # left arm
             -0.216932537318,-0.0103879115072,-0.00766682359528,0.113103572314,-0.0969100285005,0.00269548800284,-0.37497840957]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0665829935984,-0.0665829935984, # torso down
             -0.278095691606,0.0860450664571,-0.110569346238,0.539197024013,0.155210576826,-0.368374037127,-0.190716149707, # left arm
             -0.209874781044,0.206511987895,-0.00675332812243,0.117777481583,-0.0982757242583,0.00229069057572,-0.376760601537]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0662654988032,-0.0662654988032, # torso down
             -0.277398792347,0.0861068194004,-0.110443134264,0.539215528287,0.15511725643,-0.3682822681,-0.190703798986, # left arm
             -0.156316925421,0.44470963784,-0.00664687796066,0.125153532719,0.321856010043,0.0278181459061,-0.42397839986]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0662277625171,-0.0662277625171, # torso down
             -0.278366431126,0.0860735459586,-0.110624861032,0.539346473291,0.155135442635,-0.368285659934,-0.190769079708, # left arm
             -0.112253310633,0.562959936595,-0.00432966576833,0.211764412765,0.642297327417,0.800026679453,-1.16639115831]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0665408263723,-0.0665408263723, # torso down
             -0.278650471999,0.0860465392897,-0.110557703917,0.539072627537,0.154992860129,-0.36849164745,-0.190715059698, # left arm
             -0.203101995365,0.400815469071,0.047484790367,0.457016713135,0.561260833164,0.792066674003,-1.16442527649]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0673093559409,-0.0673093559409, # torso down
             -0.278595469954,0.0860445403895,-0.110681513264,0.539253364351,0.155079555031,-0.368149827378,-0.19064574711, # left arm
             -0.348776901496,0.314534641288,0.0669378486409,0.968333733284,0.762439825498,0.806605068952,-1.16816173159]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0674290759179,-0.0674290759179, # torso down
             -0.278037247633,0.085961182623,-0.110522683602,0.538803462136,0.154953035274,-0.368404503001,-0.190455923515, # left arm
             -0.303291943981,0.20477930505,0.0632472701821,1.38622592126,0.574030677567,0.768867431841,-1.15457393406]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0667404455244,-0.0667404455244, # torso down
             -0.278922701364,0.0860971260155,-0.110619882288,0.538958588653,0.155160557122,-0.368502301379,-0.190664350955, # left arm
             -0.0233446561236,-0.0582434187689,0.0269424655034,1.2229556571,0.326192645218,0.527608696793,-0.924460085333]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0667276521842,-0.0667276521842, # torso down
             -0.277683579145,0.0861586574036,-0.110471790565,0.539390154389,0.155166348183,-0.368521180424,-0.190698523268, # left arm
             0.0544393202611,-0.0839134910315,-0.157251601709,1.23228260997,0.332029684452,0.0459197131084,-0.404239017535]) # right arm
        


    # ==============================================================================================
    # Define the GoTOLocation trajectory
        self.trajGoLocation = Trajectory.Trajectory("GoToLocation", TIME_GO_TO_LOCATION)

	'''
        ==== END POSITION OF LAST ACTION ====
        self.trajGoGrasp.addPostureWP([-0.0326800201648,-0.0326800201648, # torso down
             -0.0618155402256,0.0102851437315,-0.0252052788739,0.114671676562,0.0467629891621,-0.0323641577321,-0.0368239745189, # left arm
             0.172226532381,-0.128085166684,-0.288259357413,1.09227174884,0.260798872242,0.0228264644753,-0.35851360908]) # right arm
        self.trajGoGrasp.addRHCartWP([0.3080369181313972, -0.0933579900905777, 1.01059794106796])
        self.trajGoGrasp.addRHOrientWP([-0.33079164055278293, 0.7242129795026276, 0.108525588846752, 0.595243351433504])
        ==== END POSITION OF LAST ACTION ====
	'''

        # These are the initial values as specified in the YAML ControlIt! configuration file
        #self.trajGoLocation.setInitRHCartWP([0.033912978219317776, -0.29726881641499886, 0.82])
        #self.trajGoLocation.setInitRHCartWP([0.3080369181313972, -0.0933579900905777, 1.01059794106796])
        # self.trajGoLocation.setInitRHCartWP(obj_pos)
        # self.trajGoLocation.setInitLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        # #self.trajGoLocation.setInitRHOrientWP([0.0, 1.0, 0.0, 0.0])
        # #self.trajGoLocation.setInitRHOrientWP([-0.33079164055278293, 0.7242129795026276, 0.108525588846752, 0.595243351433504])
        # self.trajGoLocation.setInitRHOrientWP(obj_ori)
        # self.trajGoLocation.setInitLHOrientWP([0.0, 1.0, 0.0, 0.0])
        # #self.trajGoLocation.setInitPostureWP(DEFAULT_POSTURE)
        # self.trajGoGrasp.addPostureWP([-0.0326800201648,-0.0326800201648, # torso down
        #      -0.0618155402256,0.0102851437315,-0.0252052788739,0.114671676562,0.0467629891621,-0.0323641577321,-0.0368239745189, # left arm
        #      0.172226532381,-0.128085166684,-0.288259357413,1.09227174884,0.260798872242,0.0228264644753,-0.35851360908]) # right arm
   
        self.trajGoLocation.setPrevTraj(self.trajGoGrasp)

        # right arm Position
        self.trajGoLocation.addRHCartWP([0.28883824418290543, -0.14405629503939205, 0.999584811185456])
        self.trajGoLocation.addRHCartWP([0.2772515572866687, -0.06550500032876867, 1.015580822684924])
        self.trajGoLocation.addRHCartWP([0.27521028676187226, -0.012900374916451662, 1.041249066881016])
        self.trajGoLocation.addRHCartWP([0.2716053128570153, -0.020591395546707494, 1.00369923204009])

        # right arm Orientation
        self.trajGoLocation.addRHOrientWP([-0.3376460043882763, 0.6715982043955052, 0.08237679282442024, 0.654343252106513])
        self.trajGoLocation.addRHOrientWP([-0.4475142222321699, 0.6072542882881371, 0.18322193332655326, 0.630399058853742])
        self.trajGoLocation.addRHOrientWP([-0.4783712340814896, 0.57760352868288, 0.2755212230296299, 0.601351130136707])
        self.trajGoLocation.addRHOrientWP([-0.47311275134691994, 0.655005207183941, 0.2203445148265389, 0.54642547329048])


        # left arm Position does not move
        self.trajGoLocation.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoLocation.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoLocation.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoLocation.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])


        # left arm Orientation does not move
        self.trajGoLocation.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoLocation.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoLocation.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoLocation.addLHOrientWP([0.0, 1.0, 0.0, 0.0])


        # trajectory
        self.trajGoLocation.addPostureWP([-0.0663117345843,-0.0663117345843, # torso down
             -0.278287560766,0.086067037334,-0.110558948578,0.539100025607,0.155069011802,-0.368305953131,-0.190699397664, # left arm
             0.0798877599587,-0.0840737354825,-0.152953479153,1.10580355632,0.310863271514,0.157436091974,-0.506464192244]) # right arm
        self.trajGoLocation.addPostureWP([-0.0662534533853,-0.0662534533853, # torso down
             -0.277350771609,0.0860679282623,-0.110472109103,0.539041581354,0.155074405309,-0.368290884565,-0.190794939774, # left arm
             0.109102862192,-0.11244546775,-0.431956221484,1.11494694164,0.385903288468,0.198866641767,-0.572591642859]) # right arm
        self.trajGoLocation.addPostureWP([-0.0656793929371,-0.0656793929371, # torso down
             -0.277188133885,0.0859684895316,-0.110772546359,0.539044921928,0.155028556167,-0.368386520124,-0.190593082396, # left arm
             0.20051037886,-0.124479721422,-0.65585392054,1.10829452975,0.395592728694,0.173719618292,-0.535368903705]) # right arm
        self.trajGoLocation.addPostureWP([-0.0656571719253,-0.0656571719253, # torso down
             -0.277395383862,0.0861746292738,-0.110544851512,0.539151624809,0.155080092715,-0.368286019862,-0.190575150443, # left arm
             0.199906896463,-0.124447569203,-0.655740456969,0.970132249503,0.485733308607,0.131035214385,-0.491288488393]) # right arm
        
        # ==============================================================================================
        #Define the GoToIdle trajectory
        self.trajGoIdle = Trajectory.Trajectory("GoToIdle", TIME_GO_TO_IDLE)

        # # These are the initial values as specified in the YAML ControlIt! configuration file
        # #self.trajGoIdle.setInitRHCartWP([0.033912978219317776, -0.29726881641499886, 0.82])
        # #self.trajGoIdle.setInitRHOrientWP([0.0, 1.0, 0.0, 0.0])
        # self.trajGoIdle.setInitRHOrientWP([-0.3417090583413905, 0.6227374186730135, 0.18086441069396716, 0.680236055921939])
        # self.trajGoIdle.setInitRHCartWP([0.2711818442848606, -0.11725385932626933, 1.06368001058201])
        # self.trajGoIdle.setInitLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        # self.trajGoIdle.setInitLHOrientWP([0.0, 1.0, 0.0, 0.0])
        # #self.trajGoIdle.setInitPostureWP(DEFAULT_POSTURE)
        # self.trajGoIdle.setInitPostureWP([-0.0332891145262,-0.0332891145262, # torso down
        #      -0.0622898657978,0.010319632391,-0.0252854737271,0.114916347834,0.0467076603524,-0.0324222505344,-0.0370686454575, # left arm
        #      0.0307719078756,0.00180269585053,-0.362223883914,1.47969587707,0.266268723735,0.0766932359528,-0.411311617134]) # right arm

        self.trajGoIdle.setPrevTraj(self.trajGoLocation)

        # right arm Position    
        self.trajGoIdle.addRHCartWP([0.2732797786391307, -0.011781985161585753, 1.053449248228718])
        self.trajGoIdle.addRHCartWP([0.2866419713758848, -0.13475678591282125, 1.016968588688599])
        self.trajGoIdle.addRHCartWP([0.2364098732210676, -0.24508974889203625, 0.968192491663854])
        self.trajGoIdle.addRHCartWP([0.15823706410292315, -0.33813623247638386, 0.93566112558224])
        self.trajGoIdle.addRHCartWP([0.05733644311166543, -0.370075341103974, 0.905560345526331])
        self.trajGoIdle.addRHCartWP([-0.0031595827411109215, -0.32738468811051774, 0.83698754518787])
        self.trajGoIdle.addRHCartWP([-0.052568894462058924, -0.27463208735477374, 0.80040609625637])
        self.trajGoIdle.addRHCartWP([-0.08715124957487375, -0.22696789675647344, 0.78823545634331])


        # right arm Orientation 
        self.trajGoIdle.addRHOrientWP([0.5191511749493414, -0.5530606801453738, -0.24438924335579507, -0.604052844838791])
        self.trajGoIdle.addRHOrientWP([-0.46962201425932115, 0.4806107095912671, 0.199305869613428, 0.713264102480162])
        self.trajGoIdle.addRHOrientWP([-0.3693163619699166, 0.4780650050047376, 0.1499428451645672, 0.782672612881708])
        self.trajGoIdle.addRHOrientWP([-0.22679365149735858, 0.5016173458771365, 0.06675035337162827, 0.832159280595028])
        self.trajGoIdle.addRHOrientWP([-0.14796871443347098, 0.5852958884182097, -0.002568149839317317, 0.797199715978403])
        self.trajGoIdle.addRHOrientWP([-0.1621679226247192, 0.7295241056045406, -0.20109888382426602, 0.633289336036943])
        self.trajGoIdle.addRHOrientWP([-0.16214107265036964, 0.8128904873945442, -0.3925644854153719, 0.3985128013675498])
        self.trajGoIdle.addRHOrientWP([-0.1537214557569671, 0.8350911417412259, -0.488612600061028, 0.2006245899357714])


 
        # left arm Position does not move
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        #self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        #self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        #self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])


        # left arm Orientation does not move
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        #self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        #self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        #self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])

        # trajectory
        self.trajGoIdle.addPostureWP([-0.0658933454842,-0.0658933454842, # torso down
             -0.27726932896,0.0859560441633,-0.1105738978,0.539102918217,0.155124323905,-0.368441346217,-0.190648521287, # left arm
             0.19315663738,-0.120984460862,-0.657077118246,1.16276149382,0.495210674225,0.167789177992,-0.563579903976]) # right arm
        self.trajGoIdle.addPostureWP([-0.0664059496526,-0.0664059496526, # torso down
             -0.277594116452,0.086083220478,-0.110523428847,0.539155516181,0.154885668161,-0.368368836821,-0.190708901317, # left arm
             0.109711100239,0.0464786838164,-0.393996088082,1.16446662445,0.613996637159,0.432553239165,-0.825521223386]) # right arm
        self.trajGoIdle.addPostureWP([-0.0667583557335,-0.0667583557335, # torso down
             -0.27828872837,0.0860112201771,-0.110597243355,0.538980186479,0.15506960637,-0.368483851315,-0.190633324568, # left arm
             -0.0851957546712,0.142719537914,-0.10204379143,1.16389957858,0.703275896209,0.614684315699,-1.0070286928]) # right arm
        self.trajGoIdle.addPostureWP([-0.0668191745596,-0.0668191745596, # torso down
             -0.277616989924,0.0861100534157,-0.110455121632,0.538650536342,0.155158962432,-0.368238061377,-0.19060072644, # left arm
             -0.266120509407,0.152352526223,0.247766171553,1.14834546613,0.798597879491,0.717378944483,-1.11266498839]) # right arm
        self.trajGoIdle.addPostureWP([-0.0669997865479,-0.0669997865479, # torso down
             -0.278308708136,0.0860346985709,-0.110547338161,0.539107237914,0.155178321208,-0.368513690572,-0.190656750261, # left arm
             -0.458932654121,0.115011213128,0.440543461974,1.12536386762,0.785362147435,0.714493531282,-1.10608034612]) # right arm
        self.trajGoIdle.addPostureWP([-0.0670090973714,-0.0670090973714, # torso down
             -0.278539324143,0.0861031232014,-0.110505908326,0.539088282133,0.155116953222,-0.368267599971,-0.19084694381, # left arm
             -0.425246503101,0.0545721987857,0.442388540068,0.773963430602,0.69607075193,0.539697367894,-0.905284367816]) # right arm
        self.trajGoIdle.addPostureWP([-0.0669876810473,-0.0669876810473, # torso down
             -0.277582286456,0.0860492615745,-0.110500851816,0.538915679189,0.155183908404,-0.368344226949,-0.190571434924, # left arm
             -0.378816463076,0.0049036750284,0.438210923442,0.472029428606,0.671445895022,0.284673962644,-0.659176435363]) # right arm
        self.trajGoIdle.addPostureWP([-0.0671572217398,-0.0671572217398, # torso down
             -0.278566240185,0.0860228028407,-0.110587824492,0.538945889011,0.155259931144,-0.368483057565,-0.190552623774, # left arm
             -0.354552329343,-0.0445709484692,0.436526878184,0.283318176516,0.670772308,0.102201180003,-0.480077185228]) # right arm

    def createFSM(self):
        # define the states
        goToGraspState = TrajectoryState(self.dreamerInterface, self.trajGoGrasp)
        goToLocationState = TrajectoryState(self.dreamerInterface, self.trajGoLocation)
        goToIdleState = TrajectoryState(self.dreamerInterface, self.trajGoIdle)

        enablePowerGraspState = EnablePowerGraspState(self.dreamerInterface)
        disablePowerGraspState = DisablePowerGraspState(self.dreamerInterface)


        # wire the states into a FSM
        self.fsm = smach.StateMachine(outcomes=['exit'])
        with self.fsm:
            smach.StateMachine.add("GoToGrasp", goToGraspState,
                transitions={'done':'EnablePowerGraspState',
                             'exit':'exit'})
            smach.StateMachine.add("EnablePowerGraspState", enablePowerGraspState,
                transitions={'done':'GoToDropLocation',
                             'exit':'exit'})
            smach.StateMachine.add("GoToDropLocation", goToLocationState,
                transitions={'done':'DisablePowerGraspState',
                             'exit':'exit'})
            smach.StateMachine.add("DisablePowerGraspState", disablePowerGraspState,
                transitions={'done':'GoToIdle',
                             'exit':'exit'})
            smach.StateMachine.add("GoToIdle", goToIdleState,
                transitions={'done':'exit',
                             'exit':'exit'})




    def run(self):

        if not self.dreamerInterface.connectToControlIt(DEFAULT_POSTURE):
            return

        self.createTrajectories()
        self.createFSM()

        # Create and start the introspection server
        sis = smach_ros.IntrospectionServer('server_name', self.fsm, '/SM_ROOT')
        sis.start()

        index = raw_input("Start demo? Y/n\n")
        if index == "N" or index == "n":
            return

        outcome = self.fsm.execute()

        print "Test trajectory done, waiting until ctrl+c is hit..."
        rospy.spin()  # just to prevent this node from exiting
        sis.stop()


# Main method
if __name__ == "__main__":
    rospy.init_node('TestTrajectory', anonymous=True)
    demo = TestTrajectory()
    demo.run()
