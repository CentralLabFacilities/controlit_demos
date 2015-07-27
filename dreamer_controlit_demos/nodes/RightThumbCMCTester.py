#!/usr/bin/env python

import rospy

import time
import math
import threading

from std_msgs.msg import Int32
from std_msgs.msg import Float64

POSITION_CONTROL_MODE = 1
DEFAULT_KP = 3
DEFAULT_KD = 0
DEFAULT_POS = 0

class SineWaveTrajGen:
    def __init__(self, publisher):
        self.publisher = publisher
        self.enabled = False

    def isRunning(self):
        return self.enabled

    def stop(self):
        self.enabled = False

    def start(self):
        self.enabled = True

        FREQ_HZ = 0.1
        AMPLITUDE = (1.57 + 0.7) / 2
        OFFSET = AMPLITUDE - 0.7
        UPDATE_PERIOD = 1.0 / 50.0


        posMsg = Float64()
        posMsg.data = DEFAULT_POS

        startTime = time.time()

        while not rospy.is_shutdown() and self.enabled:
            elapsedTime = time.time() - startTime
            goal = AMPLITUDE * math.sin(elapsedTime * 2 * math.pi * FREQ_HZ) + OFFSET

            posMsg.data = goal

            self.publisher.publish(posMsg)
            rospy.sleep(UPDATE_PERIOD)

# Main method
if __name__ == "__main__":
    rospy.init_node('Right_Thumb_CMC_Tester', anonymous=True)
    
    modePublisher = rospy.Publisher("/dreamer_controller/controlit/rightHand/mode", Int32, queue_size=1)
    kpPublisher = rospy.Publisher("/dreamer_controller/controlit/rightHand/thumb/kp", Float64, queue_size=1)
    kdPublisher = rospy.Publisher("/dreamer_controller/controlit/rightHand/thumb/kd", Float64, queue_size=1)
    posPublisher = rospy.Publisher("/dreamer_controller/controlit/rightHand/thumb/position", Float64, queue_size=1)

    posModeMsg = Int32()
    posModeMsg.data = POSITION_CONTROL_MODE

    kpMsg = Float64()
    kpMsg.data = DEFAULT_KP

    kdMsg = Float64()
    kdMsg.data = DEFAULT_KD

    posMsg = Float64()
    posMsg.data = DEFAULT_POS

    sineWaveGenerator = SineWaveTrajGen(posPublisher)
    
    options = "Please select an option:\n"\
              "  - 1: switch to position control mode\n"\
              "  - 2: set Kp gain\n"\
              "  - 3: set Kd gain\n"\
              "  - 4: set goal position\n"\
              "  - 5: toggle sine save trajectory\n"\
              "  - q: quit\n"

    done = False
    while not done:
        response = raw_input(options)

        if "q" == response:
            done = True
        elif "1" == response:
            print "RightThumbCMCTester: Setting control mode to be position..."
            modePublisher.publish(posModeMsg)
        elif "2" == response:
            response = raw_input("Enter new Kp: ")
            try:
                kpMsg.data = float(response)
                kpPublisher.publish(kpMsg)
            except ValueError:
                print "RightThumbCMCTester: Invalid Kp {0}".format(response)
        elif "3" == response:
            response = raw_input("Enter new Kd: ")
            try:
                kdMsg.data = float(response)
                kdPublisher.publish(kdMsg)
            except ValueError:
                print "RightThumbCMCTester: Invalid Kd {0}".format(response)
        elif "4" == response:
            response = raw_input("Enter new position in degrees: ")
            try:
                posDeg = float(response)
                if posDeg >= -40 and posDeg <= 90:
                    posMsg.data = posDeg / 180.0 * math.pi
                    posPublisher.publish(posMsg)
                else:
                    print "RightThumbCMCTester: Invalid position of {0}. Must be between -40 and 90 degrees.".format(posMsg.data)
            except ValueError:
                print "RightThumbCMCTester: Invalid position {0}".format(response)
        elif "5" == response:
            if sineWaveGenerator.isRunning():
                print "RightThumbCMCTester: Stopping sine wave trajectory..."
                sineWaveGenerator.stop()
            else:
                print "RightThumbCMCTester: Starting sine wave trajectory..."
                tt = threading.Thread(target=sineWaveGenerator.start)
                tt.start()
        else:
            print "RightThumbCMCTester: Unknown command {0}".format(response)


    print "RightThumbCMCTester: Done. Waiting until ctrl+c is hit..."
    rospy.spin()  # just to prevent this node from exiting