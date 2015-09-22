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
        self.trajGoGrasp.addRHCartWP([-0.11862297318999945, -0.18801605825294904, 0.77361405666206])
        self.trajGoGrasp.addRHCartWP([-0.06507384319506516, -0.3059293893078264, 0.786645195318349])
        self.trajGoGrasp.addRHCartWP([-0.04213611791555686, -0.39604147476863294, 0.819873562298347])
        self.trajGoGrasp.addRHCartWP([0.027414039096532398, -0.3583699675391953, 0.865172749888814])
        self.trajGoGrasp.addRHCartWP([0.12010629677485153, -0.35407755772768285, 1.00203633983746])
        self.trajGoGrasp.addRHCartWP([0.22010227230066762, -0.22182207447073568, 0.99246754479471])
        self.trajGoGrasp.addRHCartWP([0.2779508474888719, -0.14789014221120278, 1.036161041931661])
        self.trajGoGrasp.addRHCartWP([0.27093228416413884, -0.14985341777987818, 1.005356300608372])
        #self.trajGoGrasp.addRHCartWP([0.28257295621972356, -0.1276974132610801, 1.042172397486952])
        self.trajGoGrasp.addRHCartWP(obj_pos)


        # right arm Orientation	
        self.trajGoGrasp.addRHOrientWP([0.060571876288614684, 0.993042650005323, 0.08395397519253706, 0.056116602914406])
        self.trajGoGrasp.addRHOrientWP([0.1700519783200396, 0.9770831951108253, 0.08612594905688623, 0.0947263184152755])
        self.trajGoGrasp.addRHOrientWP([0.05068792001643176, 0.8254054756443113, 0.20031236468501448, 0.525367958763278])
        self.trajGoGrasp.addRHOrientWP([0.010534816677731849, 0.7103225398763898, 0.23489397957294011, 0.663442330079644])
        self.trajGoGrasp.addRHOrientWP([0.06740542314271032, 0.43327116595430176, 0.2914899210076928, 0.850156592419585])
        self.trajGoGrasp.addRHOrientWP([-0.06156547607049599, 0.38321592411690697, 0.2622407687555017, 0.88350723079212])
        self.trajGoGrasp.addRHOrientWP([-0.1192284345923471, 0.6657128455629978, 0.0006759437207352538, 0.7366210224648])
        self.trajGoGrasp.addRHOrientWP([-0.11267243341172979, 0.7127709978331377, -0.003215832900483932, 0.692280352036506])
        #self.trajGoGrasp.addRHOrientWP([-0.09177663006734965, 0.6953290740040672, 0.014392028585340924, 0.712662191035364])
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
        self.trajGoGrasp.addPostureWP([-0.146778015519,-0.146778015519, # torso down
             -0.376121335842,0.0158638201996,-0.103719004323,0.483782467938,0.154281234625,-0.398892756038,-0.18172053801, # left arm
             -0.357201919798,-0.0503773606935,-0.0863660413059,-0.000833842652386,-0.0980478064839,0.155911085731,0.192541638307]) # right arm
        self.trajGoGrasp.addPostureWP([-0.146471188329,-0.146471188329, # torso down
             -0.375026283126,0.0158929047873,-0.103801493469,0.4836630411,0.154363297019,-0.398853203505,-0.181642370039, # left arm
             -0.261750359008,0.163395300049,-0.0855028650657,0.00287805639355,-0.0948381232279,0.153971752289,0.191977684013]) # right arm
        self.trajGoGrasp.addPostureWP([-0.145892688642,-0.145892688642, # torso down
             -0.375383550746,0.0160621694005,-0.103980732513,0.483790784535,0.154294982829,-0.398930537023,-0.181581543244, # left arm
             -0.329860149055,0.346652161584,-0.0852161204564,0.21807738542,-0.074478244585,1.00435294964,-0.118887023268]) # right arm
        self.trajGoGrasp.addPostureWP([-0.146017081741,-0.146017081741, # torso down
             -0.375452524343,0.0160742956228,-0.103968507341,0.483817958951,0.154369146441,-0.398814331091,-0.181414360219, # left arm
             -0.51434983356,0.306808240496,0.0213749156072,0.856360961284,-0.0824756132708,0.912852707803,-0.213807774303]) # right arm
        self.trajGoGrasp.addPostureWP([-0.145866859636,-0.145866859636, # torso down
             -0.37493893704,0.0159743964674,-0.103893694046,0.483658714945,0.154230405965,-0.398965674686,-0.181505809087, # left arm
             -0.512899927744,0.26748016383,0.212292752827,1.45981386548,-0.0866757602302,1.00672673317,-0.330020192417]) # right arm
        self.trajGoGrasp.addPostureWP([-0.145447680579,-0.145447680579, # torso down
             -0.374929952414,0.016044349686,-0.103785812842,0.483787537902,0.154169025456,-0.398980573481,-0.181536137097, # left arm
             -0.173130792281,0.0566729958334,-0.0590181830182,1.22441859495,-0.153511656387,0.995727594881,-0.326948978886]) # right arm
        self.trajGoGrasp.addPostureWP([-0.145035284875,-0.145035284875, # torso down
             -0.375124666309,0.0160362688138,-0.103576528965,0.483866862984,0.154260843331,-0.398944558392,-0.181328504378, # left arm
             0.0218529006077,-0.131388952133,-0.0655486079681,1.14538095181,0.0631973378723,0.206525543559,-0.0825715107747]) # right arm
        self.trajGoGrasp.addPostureWP([-0.145165582157,-0.145165582157, # torso down
             -0.375011559269,0.0160441567252,-0.103721155766,0.483881394941,0.154498926159,-0.399046570999,-0.181621020969, # left arm
             0.0139799881599,-0.121893743584,-0.0637654949442,1.04415025249,0.0693060807364,0.187536915196,-0.0702630580702]) # right arm
        self.trajGoGrasp.addPostureWP([-0.145285549641,-0.145285549641, # torso down
             -0.374856098597,0.0158910871241,-0.103928172035,0.483879483861,0.154126646477,-0.399000132262,-0.181593453793, # left arm
             0.0422257157987,-0.177082652769,-0.0729320082579,1.13087166809,-0.0348954463531,0.121481302359,-0.0210817096094]) # right arm



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
        self.trajGoLocation.addRHCartWP([0.30475183743465306, -0.06184807309093724, 1.020398714786668])
        self.trajGoLocation.addRHCartWP([0.2983470409897317, -0.019708455414059252, 1.033503299798447])
        self.trajGoLocation.addRHCartWP([0.2972991607114904, 0.0012608185174918308, 1.0453205760794])
        self.trajGoLocation.addRHCartWP([0.2711818442848606, -0.11725385932626933, 1.06368001058201])

        # right arm Orientation
        self.trajGoLocation.addRHOrientWP([-0.3560891324029704, 0.708558195275056, 0.1522687964101584, 0.589881367168735])
        self.trajGoLocation.addRHOrientWP([-0.37676300598651663, 0.6899041674510716, 0.23293347575428885, 0.572559056278628])
        self.trajGoLocation.addRHOrientWP([-0.383084237123774, 0.6780754065752537, 0.265588281335232, 0.56826320933393])
        self.trajGoLocation.addRHOrientWP([-0.3417090583413905, 0.6227374186730135, 0.18086441069396716, 0.680236055921939])


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
        self.trajGoLocation.addPostureWP([-0.0327019704198,-0.0327019704198, # torso down
             -0.0619716798256,0.0101156905342,-0.0253527867185,0.114569140866,0.0468074614939,-0.0322958297587,-0.0369679770039, # left arm
             0.192457304398,-0.144948510281,-0.394340630979,1.09334053068,0.26523297909,0.0241629009534,-0.35392211703]) # right arm
        self.trajGoLocation.addPostureWP([-0.0323570645719,-0.0323570645719, # torso down
             -0.0617746140139,0.0101840432819,-0.0253096181008,0.114879981876,0.0468909535351,-0.0323735363804,-0.0369759619999, # left arm
             0.244619886737,-0.152589152984,-0.574792263766,1.07750006016,0.268203221481,0.0116961018213,-0.334123013401]) # right arm
        self.trajGoLocation.addPostureWP([-0.032048973966,-0.032048973966, # torso down
             -0.0626805988961,0.0100390479886,-0.0251061674822,0.114783204454,0.0468248772294,-0.0322130564779,-0.0371329202316, # left arm
             0.280300215457,-0.16839227199,-0.649080340274,1.07464832916,0.269626392731,-0.00443426807867,-0.316976433527]) # right arm
        self.trajGoLocation.addPostureWP([-0.0332891145262,-0.0332891145262, # torso down
             -0.0622898657978,0.010319632391,-0.0252854737271,0.114916347834,0.0467076603524,-0.0324222505344,-0.0370686454575, # left arm
             0.0307719078756,0.00180269585053,-0.362223883914,1.47969587707,0.266268723735,0.0766932359528,-0.411311617134]) # right arm

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
        self.trajGoIdle.addRHCartWP([0.24293794905455632, -0.20597279559217058, 1.07180906059818])
        self.trajGoIdle.addRHCartWP([0.21324284784186212, -0.27447226867494895, 1.085939221942222])
        self.trajGoIdle.addRHCartWP([0.17267863230918104, -0.3436521052516282, 1.094347737563568])
        self.trajGoIdle.addRHCartWP([0.1382529263199503, -0.37811599544977736, 1.08110660706049])
        self.trajGoIdle.addRHCartWP([0.12151434974989363, -0.39759025068198695, 1.064496723281937])
        self.trajGoIdle.addRHCartWP([0.10366475902860475, -0.38787358025880614, 0.998842057638733])
        self.trajGoIdle.addRHCartWP([0.09051218452460626, -0.38025774698972226, 0.957144873990477])
        self.trajGoIdle.addRHCartWP([0.04655529464468108, -0.35073638769124676, 0.881889942471715])
        self.trajGoIdle.addRHCartWP([-0.026464928685756832, -0.286365692393781, 0.81236217426146])
        self.trajGoIdle.addRHCartWP([-0.05907095279152361, -0.2558797840015289, 0.798189428152449])
        self.trajGoIdle.addRHCartWP([-0.035231307878553976, -0.21757572772547362, 0.78647845515626])

        # right arm Orientation 
        self.trajGoIdle.addRHOrientWP([-0.24730813619270603, 0.6180665464052947, 0.1031931991622786, 0.739042349012677])
        self.trajGoIdle.addRHOrientWP([-0.14790321619703828, 0.600347620103547, 0.04919829279262993, 0.784402257557224])
        self.trajGoIdle.addRHOrientWP([-0.05857741298827777, 0.5627069775993461, -0.04973064439417752, 0.823077400404572])
        self.trajGoIdle.addRHOrientWP([0.013936311836683027, 0.5824439466377557, -0.10233286860578913, 0.806283332483275])
        self.trajGoIdle.addRHOrientWP([0.0471884193305425, 0.6018063035227774, -0.1380725267253085, 0.78519959467961])
        self.trajGoIdle.addRHOrientWP([0.02627134336653873, 0.6642908953745903, -0.21845815775407013, 0.71435527306154])
        self.trajGoIdle.addRHOrientWP([0.005106995134426298, 0.7033072173000434, -0.2600962884546257, 0.661575995201227])
        self.trajGoIdle.addRHOrientWP([-0.02957626739436661, 0.7701931008922999, -0.35064203410054706, 0.531956761087545])
        self.trajGoIdle.addRHOrientWP([-0.06193799401334147, 0.8301516814375076, -0.4803533335212103, 0.276174846210046])
        self.trajGoIdle.addRHOrientWP([-0.07690236972549436, 0.8367023494111064, -0.5054440674254711, 0.19631988876559])
        self.trajGoIdle.addRHOrientWP([-0.17515851311328798, 0.9760947474368914, -0.11646412411010723, 0.05472336891419237])

 
        # left arm Position does not move
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])


        # left arm Orientation does not move
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])

        # trajectory
        self.trajGoIdle.addPostureWP([-0.0336161108317,-0.0336161108317, # torso down
             -0.0621466408795,0.0103192588152,-0.0252763274255,0.114867472697,0.0467606344515,-0.0323706979415,-0.0369667733199, # left arm
             -0.13317749318,0.0373131037987,-0.0703665998484,1.66845118706,0.264823990003,0.102399787262,-0.440325722829]) # right arm
        self.trajGoIdle.addPostureWP([-0.0339045744561,-0.0339045744561, # torso down
             -0.0625545598817,0.0102729736004,-0.0254116317229,0.11471461791,0.0468270883192,-0.0324497104067,-0.0369939810218, # left arm
             -0.224479637824,0.0268364260285,0.190575908437,1.79283076175,0.271718324029,0.130792369338,-0.469043199631]) # right arm
        self.trajGoIdle.addPostureWP([-0.0337627923879,-0.0337627923879, # torso down
             -0.0629635559833,0.0103047384575,-0.0252906085836,0.114684470976,0.0467750006518,-0.0324305710391,-0.0368974740871, # left arm
             -0.268677147976,-0.0113520239577,0.497532857607,1.85457959855,0.396681073103,0.154940985926,-0.503226393236]) # right arm
        self.trajGoIdle.addPostureWP([-0.0335354227251,-0.0335354227251, # torso down
             -0.0624736995183,0.0103122721463,-0.0251470556739,0.114436978884,0.0467082392819,-0.032480944481,-0.0371272981319, # left arm
             -0.298629051718,-0.0401950875718,0.672260998787,1.82711312796,0.392446580773,0.148220442265,-0.506822581267]) # right arm
        self.trajGoIdle.addPostureWP([-0.0334573913853,-0.0334573913853, # torso down
             -0.0629361164621,0.0101408182673,-0.0253501740384,0.114647895476,0.0468395772087,-0.0323042572886,-0.0369541541189, # left arm
             -0.299347073379,-0.0418895139362,0.758095024541,1.75805994625,0.386992426472,0.146208598222,-0.506520499179]) # right arm
        self.trajGoIdle.addPostureWP([-0.0332476481268,-0.0332476481268, # torso down
             -0.0620587352352,0.0103451392146,-0.0252516680985,0.114557458325,0.0468473346859,-0.0322368415847,-0.0372005724795, # left arm
             -0.309363659153,-0.0872768244334,0.771947317288,1.55450767419,0.376055917748,0.141123256984,-0.497908752228]) # right arm
        self.trajGoIdle.addPostureWP([-0.0332828355055,-0.0332828355055, # torso down
             -0.0620685965082,0.0102416228038,-0.0251850708519,0.114509759115,0.0467736723242,-0.0322915133852,-0.0370715068339, # left arm
             -0.309905767649,-0.0885816005086,0.769732894526,1.39594968305,0.37525870504,0.138619099754,-0.502203250232]) # right arm
        self.trajGoIdle.addPostureWP([-0.0332764361291,-0.0332764361291, # torso down
             -0.0626824853497,0.0101681260531,-0.0254019636435,0.114678853762,0.0468259785755,-0.0322363138316,-0.0369374189278, # left arm
             -0.3177681913,-0.0943215748453,0.771778142236,1.07658344949,0.375086543494,0.121451824741,-0.485459747949]) # right arm
        self.trajGoIdle.addPostureWP([-0.0335957515053,-0.0335957515053, # torso down
             -0.0630554785516,0.0102947305854,-0.0253267305394,0.114609837783,0.046630939302,-0.0323075719292,-0.0371039537513, # left arm
             -0.311045051816,-0.0930673992096,0.767057597606,0.632425786385,0.380227688688,0.00863715650893,-0.36370177348]) # right arm
        self.trajGoIdle.addPostureWP([-0.0338573728717,-0.0338573728717, # torso down
             -0.061998106303,0.0101248234452,-0.0252665042447,0.114821810958,0.0467191808783,-0.0321933397496,-0.0369264478877, # left arm
             -0.311066433319,-0.0944707886067,0.765699575641,0.46491109668,0.379208896728,0.00815423867272,-0.357058164593]) # right arm
        self.trajGoIdle.addPostureWP([-0.0333582536989,-0.0333582536989, # torso down
             -0.0628693686326,0.0102168026261,-0.0252980913169,0.114731676919,0.0464940552122,-0.0322951222831,-0.0370068828907, # left arm
             -0.19367736558,-0.00381088062622,0.059789765792,0.182605092727,0.189249306667,0.0106276223821,-0.356592357615]) # right arm


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
