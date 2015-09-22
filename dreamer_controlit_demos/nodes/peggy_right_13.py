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
TIME_GO_TO_GRASP = 10.0
TIME_TEST_TRAJECTORY = 2.0
TIME_GO_TO_LOCATION = 3.0
TIME_DROP = 2.0
TIME_GO_TO_IDLE = 10.0


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
	#if len(sys.argv) < 8:
    #        print 'Input argument is not enough! Abort!'
    #        sys.exit(1)
    #    obj_pos = map(float, sys.argv[1:4])
    #    obj_ori = map(float, sys.argv[4:])

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
        self.trajGoGrasp.addRHCartWP([-0.050481642138041315, -0.1924803432544305, 0.784676841200372])
        self.trajGoGrasp.addRHCartWP([-0.048853515665405266, -0.41663956784060335, 0.835232109012503])
        self.trajGoGrasp.addRHCartWP([-0.045294499281328324, -0.5669231209763754, 0.936555786452119])
        self.trajGoGrasp.addRHCartWP([-0.006472580852923062, -0.6545140909264655, 1.041778199033178])
        self.trajGoGrasp.addRHCartWP([0.2060671051398092, -0.6272792699351947, 1.189730642279882])
        self.trajGoGrasp.addRHCartWP([0.19869836265060675, -0.5125715735669205, 1.196982031800141])
        self.trajGoGrasp.addRHCartWP([0.18693432275810734, -0.4636514230384794, 1.112370929090073])
        self.trajGoGrasp.addRHCartWP([0.17674405020857506, -0.4957290386021669, 1.113223503258312])
        self.trajGoGrasp.addRHCartWP([0.18923030854954911, -0.400161134238265, 1.068656944456811])
        self.trajGoGrasp.addRHCartWP([0.1535850647658992, -0.2886026364683515, 0.995132571522979])
        self.trajGoGrasp.addRHCartWP([0.18646292827141073, -0.2328798033444794, 0.9977175988741])
        self.trajGoGrasp.addRHCartWP([0.1943050817329037, -0.20549511207804735, 0.994321823817604])
        #self.trajGoGrasp.addRHCartWP(obj_pos)


        # right arm Orientation	
        self.trajGoGrasp.addRHOrientWP([-0.05475013962730298, 0.9979277471093702, 0.00649087584061264, 0.0331738193584856])
        self.trajGoGrasp.addRHOrientWP([0.14772481204704457, 0.9883192364879203, 0.02175511621559002, 0.03048248041163732])
        self.trajGoGrasp.addRHOrientWP([0.3025651424489771, 0.952374606005593, 0.01898968524012409, 0.03280756414308648])
        self.trajGoGrasp.addRHOrientWP([0.3719446021118253, 0.7118258002623992, -0.4724661378179694, 0.3629559086942137])
        self.trajGoGrasp.addRHOrientWP([0.18676020986885453, 0.6421172998202372, -0.2829680341460418, 0.68755733501493])
        self.trajGoGrasp.addRHOrientWP([0.26140320799935857, 0.4156637058258253, -0.22218350413942511, 0.842333981859399])
        self.trajGoGrasp.addRHOrientWP([0.18245087725282436, 0.5987919077791761, -0.06344313974282327, 0.777261150828614])
        self.trajGoGrasp.addRHOrientWP([0.12510177906508868, 0.5794761241103116, 0.18025538242982997, 0.784898059346509])
        self.trajGoGrasp.addRHOrientWP([-0.06580648824776858, 0.5720472825606538, 0.29432229189381576, 0.762761955733136])
        self.trajGoGrasp.addRHOrientWP([-0.2158283256249938, 0.6120459940995164, 0.17923221727813288, 0.739387345884557])
        self.trajGoGrasp.addRHOrientWP([-0.15812808202130402, 0.697733755978806, -0.0031263552744922197, 0.698679712992061])
        self.trajGoGrasp.addRHOrientWP([-0.019558721106326577, 0.7380614650852733, -0.012399101820433103, 0.674335964085344])
        #self.trajGoGrasp.addRHOrientWP(obj_ori)


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
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])


        # trajectory
        self.trajGoGrasp.addPostureWP([-0.0349014452608,-0.0349014452608, # torso down
             -0.040120913166,0.032642726663,-0.0370570215773,0.123777475302,0.0609794956849,-0.0386281487421,-0.0357639231327, # left arm
             -0.108211528344,-0.0431323573327,-0.00182643849544,-0.0467528456901,-0.0120191617175,0.151761182095,-0.0651834770007]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0351200539263,-0.0351200539263, # torso down
             -0.0403208149082,0.0325908389546,-0.0368668937519,0.123862393812,0.0610955250027,-0.0387956759566,-0.0356732858678, # left arm
             -0.108796164907,0.372792146601,-0.00189972315389,-0.0472969133827,-0.0129196405771,0.152266824733,-0.0735290878661]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0349738821962,-0.0349738821962, # torso down
             -0.0412070757297,0.0325939832644,-0.0369934164749,0.123955000597,0.0610958375603,-0.0386287208867,-0.0357328665211, # left arm
             -0.110013028912,0.699909536629,0.000223415626942,-0.0464190575947,0.0162790930725,0.152540642426,-0.0851346863925]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0347114041214,-0.0347114041214, # torso down
             -0.0407170193948,0.0325508946684,-0.0370484738553,0.123770516894,0.0609903498323,-0.0385985807078,-0.0359776372574, # left arm
             -0.0496940617547,0.94646211216,0.0264883752771,0.013294967434,1.22516610156,0.173537314436,-0.0937568794641]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0345945538633,-0.0345945538633, # torso down
             -0.0397923768485,0.0326072042333,-0.0372183576414,0.124073746196,0.0612467200755,-0.0386693078017,-0.0358414372002, # left arm
             0.110616693601,1.05317339405,0.116572168772,0.767887776157,1.14788675674,0.162792131282,-0.0844252045708]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0339618718465,-0.0339618718465, # torso down
             -0.0409558050486,0.0325097807529,-0.036983666912,0.124025647033,0.0610790313664,-0.0387120360798,-0.0357862575858, # left arm
             0.135770852763,0.350394562994,0.853296651011,1.54270842568,0.315168296566,0.19943911454,-0.121394675435]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0345145119605,-0.0345145119605, # torso down
             -0.0399562844562,0.0325821759698,-0.037130696667,0.123695169325,0.0611247240955,-0.0386632926276,-0.0356309098893, # left arm
             -0.206171562438,0.382518731961,0.574013396349,1.53863155286,0.327121512528,0.240464514207,-0.182643963616]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0349897672926,-0.0349897672926, # torso down
             -0.0405741078243,0.0326503042569,-0.03710957588,0.123977337801,0.0611487160984,-0.0386447287404,-0.0357467729337, # left arm
             -0.314942594974,0.587874846409,0.488751205177,1.43518561825,0.370484859421,0.536655084431,-0.482085416852]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0345498957031,-0.0345498957031, # torso down
             -0.0400482932547,0.0326579739299,-0.0370526313072,0.124024492327,0.0609436135722,-0.0385135703729,-0.0358645425617, # left arm
             -0.363891836412,0.592451287243,0.127239689,1.54565940307,0.385997645076,0.608430086131,-0.549867563049]) # right arm
        self.trajGoGrasp.addPostureWP([-0.034649332037,-0.034649332037, # torso down
             -0.0406159319862,0.0325134347043,-0.0371625896713,0.123978011177,0.060946423857,-0.0386490139161,-0.0356193830101, # left arm
             -0.412627871568,0.368767046878,-0.109395424235,1.54380312107,0.432770967331,0.572195142521,-0.515537137409]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0350011682821,-0.0350011682821, # torso down
             -0.0413310922853,0.0326122907813,-0.0370292634932,0.123891668337,0.0610246826627,-0.0385863566563,-0.03572179145, # left arm
             -0.302375617358,0.146510718998,-0.09141505433,1.53056398688,0.363378476307,0.273718563776,-0.198687033933]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0345833993655,-0.0345833993655, # torso down
             -0.0405666611524,0.0326049818091,-0.0370439280914,0.123777850415,0.0610150764455,-0.0387545719745,-0.0356887219324, # left arm
             -0.272498074498,0.0496910861735,-0.0910098330405,1.5067910057,0.0848450528276,0.184473364191,0.0682976681683]) # right arm
        

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
        self.trajGoLocation.addRHCartWP([0.20449948887226677, -0.20948957210456626, 1.110838964113786])
        self.trajGoLocation.addRHCartWP([0.18046087910581973, -0.12616919149374142, 1.0977443345495])
        self.trajGoLocation.addRHCartWP([0.18926321706100066, -0.12145223903886486, 1.020784914017576])
        self.trajGoLocation.addRHCartWP([0.20192896676298455, -0.12921288071480855, 0.99309761820511])
        
        # right arm Orientation
        self.trajGoLocation.addRHOrientWP([-0.0376407795858349, 0.6054625959647735, 0.017750957077122514, 0.794784952123915])
        self.trajGoLocation.addRHOrientWP([-0.14066474540191656, 0.6110313040996426, 0.12757551832285424, 0.768491159308914])
        self.trajGoLocation.addRHOrientWP([-0.11594049292493272, 0.7000383298679741, 0.1799950555299789, 0.681253197277444])
        self.trajGoLocation.addRHOrientWP([-0.09890769038515215, 0.7304672920003714, 0.1853055230246751, 0.649843571359192])


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
        self.trajGoLocation.addPostureWP([-0.0347624942598,-0.0347624942598, # torso down
             -0.0407696996257,0.0326417990446,-0.0371064791657,0.123777453542,0.061098271116,-0.038446811658,-0.0357243759958, # left arm
             -0.259891212719,0.0895713031901,-0.0879624082974,1.91719530325,0.0796745650036,0.121841943646,0.0352053123559]) # right arm
        self.trajGoLocation.addPostureWP([-0.0350361733095,-0.0350361733095, # torso down
             -0.041726178668,0.0326195074871,-0.0370895495974,0.123418961899,0.0610157367814,-0.0387867304878,-0.0355807997238, # left arm
             -0.271498787932,0.0872719435067,-0.410784011647,1.88917980127,-0.0335515977597,0.122061093967,0.0355879385164]) # right arm
        self.trajGoLocation.addPostureWP([-0.0348512513022,-0.0348512513022, # torso down
             -0.0403793276396,0.0326711782364,-0.0370230780832,0.124099681556,0.0609643050087,-0.0384627093953,-0.0357502982058, # left arm
             -0.219156644198,0.0895702524116,-0.43949100394,1.57238614075,-0.0655707212456,0.123876532,0.0347084864628]) # right arm
        self.trajGoLocation.addPostureWP([-0.0346021902322,-0.0346021902322, # torso down
             -0.0411943398595,0.0326231408419,-0.0368738284792,0.123673435785,0.0609701631141,-0.0386208809603,-0.0359758723907, # left arm
             -0.16137803856,0.0915041318105,-0.430660772042,1.42454059265,-0.0488862084445,0.128783479435,0.037967981243]) # right arm
        
        
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
        self.trajGoIdle.addRHCartWP([0.2002293878094432, -0.12837318014228544, 1.09450408805717])
        self.trajGoIdle.addRHCartWP([0.22403461378231257, -0.20273700952067492, 1.130796315102099])
        self.trajGoIdle.addRHCartWP([0.22732227056397292, -0.2438027116706776, 1.1889611966628])
        self.trajGoIdle.addRHCartWP([0.19719140259395582, -0.28076005944485605, 1.077636954223212])
        self.trajGoIdle.addRHCartWP([0.17219922018303624, -0.3194794214282456, 1.112709053661094])
        self.trajGoIdle.addRHCartWP([0.2284038839801888, -0.44829928361815075, 1.155696354514254])
        self.trajGoIdle.addRHCartWP([0.14932392927918303, -0.6433094359761335, 1.264314997179006])
        self.trajGoIdle.addRHCartWP([0.05288667935588165, -0.6802431030883713, 1.124799854908889])
        self.trajGoIdle.addRHCartWP([-0.023142186861017676, -0.5535134542113634, 0.926477637649365])
        self.trajGoIdle.addRHCartWP([-0.03617413368521693, -0.20998774738341777, 0.78768780968166])


        # right arm Orientation 
        self.trajGoIdle.addRHOrientWP([-0.12312530433506581, 0.5977249881389018, 0.18461082095617484, 0.77037902539765])
        self.trajGoIdle.addRHOrientWP([-0.008635022506701034, 0.5596304138174526, 0.1662452644911143, 0.811850816561204])
        self.trajGoIdle.addRHOrientWP([0.0651135936484889, 0.474618929152166, 0.15830035018184652, 0.863387567170697])
        self.trajGoIdle.addRHOrientWP([0.00010204019590855813, 0.6900086439827745, 0.12535340532689765, 0.712863650770469])
        self.trajGoIdle.addRHOrientWP([0.024447278249258124, 0.6793328589574478, 0.31592758966942747, 0.661890440641477])
        self.trajGoIdle.addRHOrientWP([0.20157604118615075, 0.6796625232065113, 0.14158026565936666, 0.690927624678664])
        self.trajGoIdle.addRHOrientWP([0.46055322943055377, 0.5827798457090784, -0.12144071092349171, 0.658415163879741])
        self.trajGoIdle.addRHOrientWP([0.4682335993234306, 0.7134756199277661, -0.27530990058771226, 0.442622067764022])
        self.trajGoIdle.addRHOrientWP([0.34769445913830066, 0.8878660121676856, -0.22085236421132523, 0.2050042456737029])
        self.trajGoIdle.addRHOrientWP([0.025172530517963726, 0.9809712267488497, -0.15582018376544834, 0.1130569163274894])
 
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

        # trajectory
        self.trajGoIdle.addPostureWP([-0.035041443134,-0.035041443134, # torso down
             -0.0407938985819,0.0327043698482,-0.0371591510346,0.123915737246,0.0609767434975,-0.0386349383146,-0.0356138545595, # left arm
             -0.179480252261,0.140838997472,-0.447038898547,1.81943263423,-0.0655767457119,0.127778324088,0.0286252274714]) # right arm
        self.trajGoIdle.addPostureWP([-0.0346013725692,-0.0346013725692, # torso down
             -0.0404276070508,0.0326096193082,-0.0370083284433,0.124028545697,0.0609984931353,-0.0384280135755,-0.0357642111359, # left arm
             -0.146027689485,0.251388358947,-0.232203949477,1.90005361554,-0.0513857933674,0.129786261731,0.0390046186706]) # right arm
        self.trajGoIdle.addPostureWP([-0.0345444140843,-0.0345444140843, # torso down
             -0.0408204666527,0.0325550443775,-0.0370701716297,0.123773549571,0.0609623286641,-0.0385800715487,-0.035672170807, # left arm
             -0.0993800251952,0.317442107887,-0.0809764081246,2.03134614125,-0.0515803320387,0.137990266046,0.0364327115095]) # right arm
        self.trajGoIdle.addPostureWP([-0.0350942651603,-0.0350942651603, # torso down
             -0.0404285738926,0.0327091768615,-0.0369835681776,0.12370366762,0.0608421595783,-0.0386801051297,-0.0357241439553, # left arm
             -0.311628719567,0.456048093946,-0.12624257182,1.76653735742,0.226369950121,0.12722600649,0.0430162324847]) # right arm
        self.trajGoIdle.addPostureWP([-0.0349197633609,-0.0349197633609, # torso down
             -0.0413138722786,0.0328072487967,-0.0370163224079,0.123843439849,0.0611263177167,-0.0387616592708,-0.0357010561678, # left arm
             -0.456388219065,0.842630031907,-0.168616756594,1.88598448024,0.214888123908,0.153789352191,0.0352335861887]) # right arm
        self.trajGoIdle.addPostureWP([-0.0343278810064,-0.0343278810064, # torso down
             -0.0413880236935,0.0326048016834,-0.0371658593811,0.123923651384,0.0610555078597,-0.0385485555454,-0.0356854405447, # left arm
             -0.256418718685,0.849020261651,0.145856509265,1.53980822332,0.371534872788,0.137581085806,0.0263760834605]) # right arm
        self.trajGoIdle.addPostureWP([-0.0341463515117,-0.0341463515117, # torso down
             -0.0404569307052,0.0325994245575,-0.0370695634867,0.123774989011,0.0610809131643,-0.0386458581761,-0.0358773650739, # left arm
             -0.195047807143,1.00106466904,0.761624417157,1.05482685768,0.375881337508,0.137297918564,0.0264710183834]) # right arm
        self.trajGoIdle.addPostureWP([-0.0341683006142,-0.0341683006142, # torso down
             -0.0406763568062,0.0325935831562,-0.0369652797727,0.123877272408,0.0611546940425,-0.038423034386,-0.0357614731687, # left arm
             -0.224044349834,0.948675644312,0.719916775619,0.513043161672,0.37622555557,0.136723503909,0.0232947679167]) # right arm
        self.trajGoIdle.addPostureWP([-0.0345810799983,-0.0345810799983, # torso down
             -0.0408206111579,0.0327227355273,-0.036932288479,0.124126768744,0.0609640517321,-0.0385643170974,-0.0357349031189, # left arm
             -0.247007706696,0.620736809446,0.411472164973,0.264743923635,0.254647726822,0.141811527278,0.0286762958092]) # right arm
        self.trajGoIdle.addPostureWP([-0.0348973213927,-0.0348973213927, # torso down
             -0.0413644317422,0.0326808167838,-0.0370411723782,0.123833236007,0.0609405463001,-0.0385111206705,-0.0358528466127, # left arm
             -0.234308607784,-0.0367051476743,0.178780095273,0.262349686097,0.127307862519,0.139309280536,0.0334679656265]) # right arm


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
