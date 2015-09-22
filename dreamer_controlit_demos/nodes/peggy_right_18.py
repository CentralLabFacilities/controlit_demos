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
        self.trajGoGrasp.setInitLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.setInitRHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.setInitLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.setInitPostureWP(DEFAULT_POSTURE)




        self.trajGoGrasp.addRHCartWP([-0.05704398087886885, -0.3221287909375606, 0.797766985177679])
        self.trajGoGrasp.addRHCartWP([-0.05786166530387322, -0.40220703807953484, 0.824792954741813])
        self.trajGoGrasp.addRHCartWP([-0.05828356258095457, -0.46304786872075004, 0.855187250043026])
        self.trajGoGrasp.addRHCartWP([-0.059504334236596394, -0.5116292060182731, 0.886770752665177])
        self.trajGoGrasp.addRHCartWP([-0.06000626749963865, -0.5487168023341688, 0.916176168032419])
        self.trajGoGrasp.addRHCartWP([-0.06017206601086252, -0.5747268317955826, 0.940174435762382])
        self.trajGoGrasp.addRHCartWP([-0.060346916600567985, -0.6101923195136341, 0.97847344951805])
        self.trajGoGrasp.addRHCartWP([-0.059588753037214344, -0.6343538013343242, 1.00929614490673])
        self.trajGoGrasp.addRHCartWP([-0.060064765587074534, -0.6604569855070778, 1.048473508045037])
        self.trajGoGrasp.addRHCartWP([-0.060638818259622626, -0.6779978184728963, 1.079631827394153])
        self.trajGoGrasp.addRHCartWP([-0.06074540020949129, -0.6963881532139035, 1.118453631295711])
        self.trajGoGrasp.addRHCartWP([-0.06093217589402831, -0.7061224041042282, 1.142771837600565])
        self.trajGoGrasp.addRHCartWP([0.015084259208199153, -0.7097457049743622, 1.172012875773761])
        self.trajGoGrasp.addRHCartWP([0.11038805586401963, -0.6665449633717602, 1.14834585240725])
        self.trajGoGrasp.addRHCartWP([0.15731068107890767, -0.6302791254522718, 1.147340550638468])
        self.trajGoGrasp.addRHCartWP([0.21513446620308074, -0.5579633220648338, 1.142888199342187])
        self.trajGoGrasp.addRHCartWP([0.25584347733529156, -0.4199857446100331, 1.1088252469906])
        self.trajGoGrasp.addRHCartWP([0.24604647026961143, -0.3209342052223624, 1.11001320942097])
        self.trajGoGrasp.addRHCartWP([0.2282655808844735, -0.19870041767928323, 1.108636457717073])
        self.trajGoGrasp.addRHCartWP([0.21431564421402677, -0.22455602164894004, 1.041718243595557])
        self.trajGoGrasp.addRHCartWP([0.21465169223092945, -0.19174781580978864, 1.009693490453695])



        self.trajGoGrasp.addRHOrientWP([0.23098838168268715, 0.9675793264872992, 0.0003693307182457862, 0.1021492930803644])
        self.trajGoGrasp.addRHOrientWP([0.3024953139676857, 0.9477376836210876, 0.010332538344227652, 0.1009113805449696])
        self.trajGoGrasp.addRHOrientWP([0.35863674462185735, 0.9279894417356506, 0.017592088551628162, 0.0995278848091534])
        self.trajGoGrasp.addRHOrientWP([0.406171293684226, 0.908278078937082, 0.02265077735163565, 0.0976870195809689])
        self.trajGoGrasp.addRHOrientWP([0.4440247010799266, 0.8904449991827978, 0.025633259802095975, 0.0963986735030483])
        self.trajGoGrasp.addRHOrientWP([0.47166491101080926, 0.8761061023565329, 0.027348300387266117, 0.0960332213389597])
        self.trajGoGrasp.addRHOrientWP([0.5116460730183834, 0.8533692225619521, 0.031899777791214286, 0.094666098080293])
        self.trajGoGrasp.addRHOrientWP([0.529495342320728, 0.8406560665949759, 0.038329952082675395, 0.1070648164947674])
        self.trajGoGrasp.addRHOrientWP([0.5610041350538615, 0.8193599572143567, 0.04391447665627841, 0.1095223251495389])
        self.trajGoGrasp.addRHOrientWP([0.5866897500682076, 0.8012373097656873, 0.0466689804713129, 0.107869907138721])
        self.trajGoGrasp.addRHOrientWP([0.5696474288868428, 0.8037886250446477, 0.0798239380292245, 0.1518347520462641])
        self.trajGoGrasp.addRHOrientWP([0.5566177738628546, 0.8112864784403562, 0.09239173309083226, 0.1531491801358333])
        self.trajGoGrasp.addRHOrientWP([0.4445090861494361, 0.792099066064288, 0.3138239292081926, 0.276595884508441])
        self.trajGoGrasp.addRHOrientWP([0.369569116459723, 0.7939739738227485, 0.34131580305141024, 0.34136127436876])
        self.trajGoGrasp.addRHOrientWP([0.3041441992678841, 0.7329439160482023, 0.17686153661279289, 0.582245239436601])
        self.trajGoGrasp.addRHOrientWP([0.1486728358074718, 0.6252106321619038, 0.04125709062313756, 0.76505287777904])
        self.trajGoGrasp.addRHOrientWP([-0.03167977892204318, 0.6170681358936099, 0.12334361419435423, 0.776536966351061])
        self.trajGoGrasp.addRHOrientWP([-0.07168512761139015, 0.7288004404916192, 0.08228806941488738, 0.675973249508093])
        self.trajGoGrasp.addRHOrientWP([-0.15104117299942893, 0.7385347947183847, 0.0679862322093164, 0.653552441108753])
        self.trajGoGrasp.addRHOrientWP([-0.10533527380389113, 0.7363682534182102, 0.1190790276509253, 0.657637028020916])
        self.trajGoGrasp.addRHOrientWP([-0.09168415557052449, 0.7337650921714186, 0.041276367227237244, 0.671921920044666])


        # left arm doesn't move
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
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])


        # left arm doesn't move
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
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])


        self.trajGoGrasp.addPostureWP([-0.0886034606526,-0.0886034606526, # torso down
            -0.285252307774,0.0829043658651,-0.166177818826,0.535277914506,0.198551702288,-0.429745983962,-0.173993665584, # left arm
            -0.162091058785,0.19494426858,0.110129334474,-0.0603410030915,-0.123488318503,0.249401076868,0.282379438022]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0887070133498,-0.0887070133498, # torso down
            -0.285805367061,0.0828024722589,-0.166026224877,0.535463141372,0.19867423326,-0.429735697902,-0.173985911419, # left arm
            -0.16250572622,0.346985358274,0.109465629178,-0.0608593441514,-0.129525002287,0.249282492096,0.280155703219]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0885121431152,-0.0885121431152, # torso down
            -0.285618737834,0.0829527625272,-0.166108953476,0.535506760214,0.198831819254,-0.429704768008,-0.17404172527, # left arm
            -0.162186134294,0.469350880774,0.110297538075,-0.0613641744299,-0.134135447256,0.249289308611,0.27807615231]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0882658437502,-0.0882658437502, # torso down
            -0.285535855801,0.0830607710459,-0.166145369444,0.535299013218,0.198491784737,-0.42971885239,-0.174003448476, # left arm
            -0.16406941847,0.573485669995,0.110123462389,-0.0606403294375,-0.133663953139,0.249699053333,0.276903306712]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0885906281887,-0.0885906281887, # torso down
            -0.28578917873,0.0829684586097,-0.166155353865,0.535128399811,0.198604799647,-0.430003641586,-0.174093211179, # left arm
            -0.164377735204,0.658577232577,0.110338026556,-0.0611033670218,-0.132597890495,0.249780423643,0.275761875855]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0885245540122,-0.0885245540122, # torso down
            -0.285971122738,0.0830335080449,-0.166207161854,0.535299169652,0.198694353232,-0.429875715555,-0.173984804409, # left arm
            -0.164083963197,0.722209602548,0.110758584392,-0.0608119412714,-0.131245995855,0.250152928504,0.274316175626]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0880934705561,-0.0880934705561, # torso down
            -0.285327066382,0.0830825725379,-0.166058802238,0.535391627981,0.198658986582,-0.429881393578,-0.174119468478, # left arm
            -0.16338157162,0.816065104123,0.110617599104,-0.0606218854471,-0.131944912173,0.250836450711,0.272928888238]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0881779688874,-0.0881779688874, # torso down
            -0.285338497299,0.0829288784198,-0.166186374103,0.535486198414,0.198669971687,-0.42999491512,-0.173923543919, # left arm
            -0.160138900851,0.886552779833,0.110825696389,-0.0603219361016,-0.129559148264,0.277058799299,0.245789669261]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0884318864306,-0.0884318864306, # torso down
            -0.284929576844,0.0829034860327,-0.166009658423,0.535303019004,0.198575280277,-0.429799696039,-0.174239947394, # left arm
            -0.159919152719,0.971051062637,0.110836708346,-0.0599169834708,-0.12972396595,0.28592965782,0.237321534065]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0883555763485,-0.0883555763485, # torso down
            -0.28552618033,0.0829748457975,-0.166095193804,0.535209203713,0.198610561267,-0.429857728483,-0.174042329589, # left arm
            -0.160347749071,1.03541200703,0.111220654036,-0.0596747026444,-0.128465587991,0.286792574222,0.236718767673]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0882062745701,-0.0882062745701, # torso down
            -0.285637205252,0.0830243589517,-0.165994381374,0.53548582653,0.198674309997,-0.429994297364,-0.173966833048, # left arm
            -0.159551801628,1.11244500792,0.111681282702,-0.0584376710311,-0.125201367587,0.395201272874,0.12753876352]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0884824394242,-0.0884824394242, # torso down
            -0.28534244337,0.0829823877807,-0.166075457021,0.535250168974,0.198600280494,-0.429726326624,-0.173939478751, # left arm
            -0.158744327312,1.15961058385,0.111522611371,-0.0579329508141,-0.125100756357,0.410096046002,0.0497782245209]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0886002390146,-0.0886002390146, # torso down
            -0.286505473723,0.0830910323752,-0.166098018085,0.535223003707,0.198611666652,-0.429762192347,-0.174139031219, # left arm
            -0.151799183426,1.19213650863,0.116351049481,0.217437425753,-0.271818628953,0.630034142373,-0.1720775087]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0887610528059,-0.0887610528059, # torso down
            -0.285725028813,0.0831846610979,-0.165932256753,0.535036224634,0.198657207829,-0.429859726696,-0.174063350236, # left arm
            -0.0244600986602,1.06825402174,0.112865202219,0.484244844713,-0.302211068097,0.466269137851,-0.335488018844]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0885768574465,-0.0885768574465, # torso down
            -0.285841995805,0.0830098970114,-0.166068988514,0.535376538082,0.198599989993,-0.429698258733,-0.173826854859, # left arm
            -0.017603362086,1.00686415239,0.11141571648,0.678416692605,0.243531104952,0.457028415317,-0.333585940386]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0882794735967,-0.0882794735967, # torso down
            -0.286144434268,0.0828756582028,-0.166205156599,0.535354479257,0.198667818977,-0.429748738862,-0.173784919174, # left arm
            0.00186384559962,0.875948497001,0.103890862359,0.951829113089,0.667223927109,0.452357995617,-0.324460005534]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0882728259132,-0.0882728259132, # torso down
            -0.285634382368,0.082936675246,-0.166025955854,0.53537776175,0.198661595269,-0.429754260417,-0.173927995686, # left arm
            -0.00276824871892,0.739940690905,-0.170192090476,1.25719711386,0.677484042355,0.416682146319,-0.285966656163]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0884243795673,-0.0884243795673, # torso down
            -0.286345408096,0.0830226823993,-0.166183644737,0.535345732282,0.198646240045,-0.429759325632,-0.173892148162, # left arm
            -0.011687282999,0.734577797134,-0.414956495466,1.51510265101,0.744525364143,0.080613825049,0.050810221454]) # right arm
        self.trajGoGrasp.addPostureWP([-0.088031115249,-0.088031115249, # torso down
            -0.285390080648,0.0829230879877,-0.166178140958,0.535504578117,0.198714614444,-0.429811901129,-0.174066256261, # left arm
            -0.0120386525089,0.481491643502,-0.542600909871,1.68483500383,0.649292594062,-0.0726881047631,0.22615959251]) # right arm
        self.trajGoGrasp.addPostureWP([-0.088267065525,-0.088267065525, # torso down
            -0.28605185351,0.0827697015494,-0.165998237947,0.535399571001,0.198605322493,-0.429725603248,-0.174001775969, # left arm
            -0.0855228092317,0.49535579112,-0.553082651439,1.49671257641,0.495793170191,0.125676160826,0.131724504418]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0882129531136,-0.0882129531136, # torso down
            -0.28643150677,0.0828547516782,-0.165970243724,0.535442677809,0.198720613122,-0.429728559171,-0.174105281456, # left arm
            -0.0868963277147,0.34340622311,-0.541423248911,1.37717090843,0.453493885905,0.2191790536,0.22885285511]) # right arm

    # ==============================================================================================
    # Define the GoTOLocation trajectory
        self.trajGoLocation = Trajectory.Trajectory("GoToLocation", TIME_GO_TO_LOCATION)

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
        self.trajGoIdle.addRHCartWP([0.23806006449534947, -0.16812808429318704, 1.08977442798251])
        self.trajGoIdle.addRHCartWP([0.24678239048155023, -0.2445280348210488, 1.093925295017103])
        self.trajGoIdle.addRHCartWP([0.23175663723007722, -0.34317208305306757, 1.085831091538392])
        self.trajGoIdle.addRHCartWP([0.1957634933653613, -0.4228106980352024, 1.08590972231969])
        self.trajGoIdle.addRHCartWP([0.13577429601771288, -0.4976886386836021, 1.055015127397333])
        self.trajGoIdle.addRHCartWP([0.034022985578650286, -0.5500466314741769, 1.021705335659044])
        self.trajGoIdle.addRHCartWP([-0.013072050180572442, -0.43403671618190826, 0.851903641868219])
        self.trajGoIdle.addRHCartWP([-0.046992563669959894, -0.33078964089801405, 0.802442249215622])
        self.trajGoIdle.addRHCartWP([-0.0690257483812168, -0.23790979841313592, 0.783805030222298])
        self.trajGoIdle.addRHCartWP([-0.07977168995910655, -0.26297665853730173, 0.788066034782979])
        self.trajGoIdle.addRHCartWP([-0.06938358563815773, -0.22269637578042162, 0.783576126666514])
        self.trajGoIdle.addRHCartWP([-0.06132186654886571, -0.18859770420957386, 0.782043031563492])

        # right arm orientation
        self.trajGoIdle.addRHOrientWP([-0.05804422045319227, 0.6892886456791986, 0.027319761542346006, 0.721640950916026])
        self.trajGoIdle.addRHOrientWP([0.0420907332329711, 0.6803451262341618, -0.06910491683285373, 0.728411552527043])
        self.trajGoIdle.addRHOrientWP([0.14622895901099847, 0.6628024636557794, -0.18053506440957415, 0.71184062558765])
        self.trajGoIdle.addRHOrientWP([0.261850482405012, 0.6003411654426861, -0.2770590930145522, 0.703038454792576])
        self.trajGoIdle.addRHOrientWP([0.43300177832457637, 0.5018771263307832, -0.3924482408379325, 0.63766228389087])
        self.trajGoIdle.addRHOrientWP([0.5075736779971044, 0.44556192753897006, -0.5235165625260111, 0.519397669318219])
        self.trajGoIdle.addRHOrientWP([0.39757569878534316, 0.6954334888209789, -0.5368653197077603, 0.264729021561127])
        self.trajGoIdle.addRHOrientWP([0.3300880088361215, 0.8665539036627806, -0.33955591924444484, 0.1575690838186889])
        self.trajGoIdle.addRHOrientWP([0.20920290764559468, 0.959904736921256, -0.15243802801683287, 0.1076089544745338])
        self.trajGoIdle.addRHOrientWP([0.1849490259271349, 0.9750419662749787, 0.04331816595489282, 0.1149371928904289])
        self.trajGoIdle.addRHOrientWP([0.10848498616024886, 0.9502729976186355, 0.24820287279699615, 0.1536475567980877])
        self.trajGoIdle.addRHOrientWP([0.13695786485364428, 0.9798541844580554, -0.08929968522778944, 0.1146903948593245])


        # left arm doesn't move
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
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])


        # left arm doesn't move
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
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])


        self.trajGoIdle.addPostureWP([-0.0669569934005,-0.0669569934005, # torso down
            -0.277346717644,0.0625617953671,-0.122911801737,0.545553771345,0.172198200909,-0.362218817077,-0.176389408324, # left arm
            -0.0828279393923,0.174905223639,-0.339518798998,1.65209848697,0.210827669057,-0.0140059461786,0.229339100035]) # right arm
        self.trajGoIdle.addPostureWP([-0.0668356331335,-0.0668356331335, # torso down
            -0.278059286633,0.0625583163478,-0.123033065504,0.545647683565,0.172132538926,-0.362271085326,-0.176658058859, # left arm
            -0.109102114669,0.181464974327,-0.0649566641951,1.64339461228,0.210281859941,-0.0105670855834,0.235870513922]) # right arm
        self.trajGoIdle.addPostureWP([-0.0669364413641,-0.0669364413641, # torso down
            -0.277335034852,0.0623533949709,-0.122892258994,0.54560380799,0.172138697952,-0.36237979686,-0.176377288196, # left arm
            -0.136810993524,0.20342921713,0.259670003135,1.56923090683,0.24775861969,0.0208999380518,0.212466533211]) # right arm
        self.trajGoIdle.addPostureWP([-0.0667615308683,-0.0667615308683, # torso down
            -0.278123690122,0.0625046520159,-0.122976882533,0.545738812265,0.172136298507,-0.362138311689,-0.176503482748, # left arm
            -0.117939113333,0.167777869546,0.604001566993,1.51971288173,0.180902147432,0.0803449826369,0.178614506884]) # right arm
        self.trajGoIdle.addPostureWP([-0.0670801364241,-0.0670801364241, # torso down
            -0.277922021023,0.0624457805303,-0.122979849186,0.545710016693,0.172276779691,-0.362264161415,-0.176430157406, # left arm
            -0.120214967272,0.21496108816,0.89043092093,1.32384986737,0.122913489356,0.27037365636,0.342388242262]) # right arm
        self.trajGoIdle.addPostureWP([-0.0669533149885,-0.0669533149885, # torso down
            -0.277543420791,0.0625100191736,-0.123006935487,0.545784265346,0.172084818395,-0.362339731543,-0.176624084725, # left arm
            -0.12102407016,0.243561193453,1.29177111832,1.13080160352,0.127254035661,0.282357513999,0.309297317776]) # right arm
        self.trajGoIdle.addPostureWP([-0.0673218768229,-0.0673218768229, # torso down
            -0.277213443375,0.0624860194135,-0.122847414174,0.545649967851,0.172034700149,-0.362262725587,-0.176628655393, # left arm
            -0.19277931445,0.175408584304,1.16279582674,0.511526702287,0.0499538675104,0.27049721285,0.306251859392]) # right arm
        self.trajGoIdle.addPostureWP([-0.0673558711853,-0.0673558711853, # torso down
            -0.276961492108,0.0625948694083,-0.123168401629,0.545642788456,0.172188522861,-0.362076345223,-0.176450299657, # left arm
            -0.211720465819,0.0612837535295,1.16366097415,0.305813987218,-0.450347540539,0.275658893106,0.301685071753]) # right arm
        self.trajGoIdle.addPostureWP([-0.0674228199809,-0.0674228199809, # torso down
            -0.277837520036,0.0623686270456,-0.122826114885,0.545728453527,0.172060941064,-0.36230606451,-0.17653223036, # left arm
            -0.215284492784,-0.0217748645154,1.11537911177,0.11614566853,-0.827524281504,0.279503420012,0.30277813675]) # right arm
        self.trajGoIdle.addPostureWP([-0.0675698501565,-0.0675698501565, # torso down
            -0.277816935041,0.0624014702687,-0.123088108519,0.545803707466,0.172060147599,-0.36211085995,-0.176358418516, # left arm
            -0.228575760554,0.0636863274989,0.724487861251,0.0432980502212,-0.828225137412,0.280941538462,0.301625777588]) # right arm
        self.trajGoIdle.addPostureWP([-0.0676580463937,-0.0676580463937, # torso down
            -0.277908923,0.062334910171,-0.122844839816,0.5455252224,0.172018450288,-0.362362365705,-0.176528563738, # left arm
            -0.216608393365,0.00240349729189,0.280177966864,0.0449444777162,-0.828021983447,0.280759641616,0.302878616289]) # right arm
        self.trajGoIdle.addPostureWP([-0.067634978329,-0.067634978329, # torso down
            -0.277179721789,0.0626350983513,-0.123517478015,0.545506826506,0.171819278972,-0.362064828929,-0.175687942545, # left arm
            -0.203697850154,-0.0548539569602,0.152004692106,0.0466337024757,-0.0181904042239,0.280543921912,0.304179239269]) # right arm




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