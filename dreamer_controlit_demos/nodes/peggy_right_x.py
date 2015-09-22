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
        self.trajGoGrasp.addRHCartWP([-0.04222796117255566, -0.2124789129775895, 0.784932921037725])
        self.trajGoGrasp.addRHCartWP([-0.0421518805160168, -0.3439687551476469, 0.807541560065679])
        self.trajGoGrasp.addRHCartWP([-0.08816909461651823, -0.46095252096190736, 0.861822322539531])
        self.trajGoGrasp.addRHCartWP([-0.03821711446379648, -0.5107296478839709, 0.89689877795771])
        self.trajGoGrasp.addRHCartWP([0.17887909015065387, -0.4486076348712261, 1.01708827587344])
        self.trajGoGrasp.addRHCartWP([0.17812577577466643, -0.4188360347594363, 1.04868035477958])
        self.trajGoGrasp.addRHCartWP([0.2618410334996237, -0.32940861291431167, 1.064498329638329])
        self.trajGoGrasp.addRHCartWP([0.2846468995655185, -0.24952102140542085, 1.04767977867554])
        self.trajGoGrasp.addRHCartWP([0.31106196687179394, -0.17161362029772456, 1.026370385147279])
        self.trajGoGrasp.addRHCartWP([0.3187051010994612, -0.15906456115306464, 1.000383523940843])
#        self.trajGoGrasp.addRHCartWP([0.3080369181313972, -0.0933579900905777, 1.01059794106796])
        self.trajGoGrasp.addRHCartWP(obj_pos)


        # right arm Orientation	
        self.trajGoGrasp.addRHOrientWP([-7.007738094213022e-05, 0.7139337618494572, -0.699694415967018, -0.02695001011533797])
        self.trajGoGrasp.addRHOrientWP([0.08090974479792618, 0.7112717452418571, -0.6960293338635262, 0.05558132796932])
        self.trajGoGrasp.addRHOrientWP([-0.10379079789993734, 0.7192284651912281, -0.6487218978300133, 0.2260481904501572])
        self.trajGoGrasp.addRHOrientWP([-0.38633381884050966, 0.6569550578173199, -0.5275368050802712, 0.375314736873001])
        self.trajGoGrasp.addRHOrientWP([0.631989286767991, -0.5704509613747093, 0.1694573470048214, -0.496446824567589])
        self.trajGoGrasp.addRHOrientWP([-0.4416507015483431, 0.6348559720076071, 0.055832263985717286, 0.631494505856038])
        self.trajGoGrasp.addRHOrientWP([-0.3655758467641159, 0.6713955891703223, 0.14293569542182574, 0.628610889246769])
        self.trajGoGrasp.addRHOrientWP([-0.34326038561510086, 0.6378131526435099, 0.08406804029805842, 0.684323939799539])
        self.trajGoGrasp.addRHOrientWP([-0.25142779075867455, 0.7143292496097823, 0.022791391235525416, 0.652685484495929])
        self.trajGoGrasp.addRHOrientWP([-0.26299771477353456, 0.7483194160718256, 0.017061381046223095, 0.608735708522685])
#        self.trajGoGrasp.addRHOrientWP([-0.33079164055278293, 0.7242129795026276, 0.108525588846752, 0.595243351433504])
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
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])


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
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])


        # trajectory
        self.trajGoGrasp.addPostureWP([-0.0334325634911,-0.0334325634911, # torso down
             -0.0620332990323,0.0102523815912,-0.0252667285024,0.11463355218,0.0467721341108,-0.0322313754365,-0.0369477502241, # left arm
             -0.0961914875276,-0.00792451454782,-0.0733970639045,-0.038549216767,1.62226461937,-0.0320963394618,-0.0300104806233]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0333138715567,-0.0333138715567, # torso down
             -0.0626270424963,0.0100926808244,-0.0254137676317,0.11478848456,0.0468871448904,-0.0322165455013,-0.0371169117357, # left arm
             -0.0970721912243,0.232386633828,-0.0729216674426,-0.0386866217374,1.62326262048,-0.0409346365186,-0.0357454651119]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0333742182662,-0.0333742182662, # torso down
             -0.0619527704455,0.0102087735708,-0.0252322503797,0.114835727616,0.0468023084956,-0.0322770655738,-0.0372123719557, # left arm
             -0.214500325987,0.462620220714,-0.0741879998063,-0.00624950388246,1.61518376678,-0.264664036173,-0.637409158425]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0336773397104,-0.0336773397104, # torso down
             -0.0627540854973,0.0103171769576,-0.0252175522909,0.114734013833,0.0469391223623,-0.0324198970996,-0.0368907380059, # left arm
             -0.30767050718,0.595545736893,-0.0671830583358,0.30580813178,1.6407798079,-0.470538406617,-1.11788121049]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0337954267959,-0.0337954267959, # torso down
             -0.0627979422266,0.0101687394481,-0.0253226767132,0.114344700863,0.0465851553307,-0.0326888132478,-0.0371286130182, # left arm
             -0.281473013362,0.610087766424,0.104595192807,1.21320089953,1.50553064479,-0.487824486881,-1.10864058124]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0339392248246,-0.0339392248246, # torso down
             -0.0629890762818,0.0102118736039,-0.0253281549406,0.114618422518,0.0468602644274,-0.0324334510504,-0.0370703617512, # left arm
             -0.309602669123,0.373874630149,0.363973796264,1.47506050197,0.835423837193,-0.201817181648,-1.12513213094]) # right arm
        self.trajGoGrasp.addPostureWP([-0.033120208572,-0.033120208572, # torso down
             -0.06210482469,0.0101195901825,-0.0252165676017,0.114760806188,0.0468394858785,-0.0324114381541,-0.0371120398593, # left arm
             -0.0133994895031,0.0793572768898,0.323529964532,1.48922335041,0.264659098307,-0.242497285822,-1.04960039443]) # right arm
        self.trajGoGrasp.addPostureWP([-0.033087876253,-0.033087876253, # torso down
             -0.0623107169357,0.010270498774,-0.0252249867414,0.114778136174,0.0466645469942,-0.0322484662629,-0.0370198910823, # left arm
             0.0179011694781,0.0834016628702,0.0232379157584,1.41428996933,0.449952970398,0.00295313646175,-0.674610123477]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0324967284752,-0.0324967284752, # torso down
             -0.0627004384408,0.0102324639434,-0.0251575516519,0.114773829818,0.0468066349597,-0.0324826320575,-0.0367770349735, # left arm
             0.122582765466,-0.0767888653589,-0.0612310463568,1.23370435946,0.247466154743,0.0155090397397,-0.337203143429]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0323745729622,-0.0323745729622, # torso down
             -0.0617019235675,0.0102114804234,-0.0254394774859,0.114492328508,0.0469308189417,-0.0323705384137,-0.0370808355226, # left arm
             0.174533962289,-0.0959026934672,-0.0720113229884,1.07201612531,0.249361013417,0.0125528561504,-0.338443965867]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0326800201648,-0.0326800201648, # torso down
             -0.0618155402256,0.0102851437315,-0.0252052788739,0.114671676562,0.0467629891621,-0.0323641577321,-0.0368239745189, # left arm
             0.172226532381,-0.128085166684,-0.288259357413,1.09227174884,0.260798872242,0.0228264644753,-0.35851360908]) # right arm



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
