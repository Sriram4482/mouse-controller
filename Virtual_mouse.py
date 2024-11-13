# Imports

import cv2
import mediapipe as mp
import pyautogui  # type: ignore
import math
from enum import IntEnum
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from google.protobuf.json_format import MessageToDict
import screen_brightness_control as sbcontrol

pyautogui.FAILSAFE = False

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Gesture Encodings
class Gest(IntEnum):
    FIST = 0
    PINKY = 1
    RING = 2
    MID = 4
    LAST3 = 7
    INDEX = 8
    FIRST2 = 12
    LAST4 = 15
    THUMB = 16
    PALM = 31

    # Extra Mappings
    V_GEST = 33
    TWO_FINGER_CLOSED = 34
    PINCH_MAJOR = 35
    PINCH_MINOR = 36

# Multi-handedness Labels
class HLabel(IntEnum):
    MINOR = 0
    MAJOR = 1

# Convert Mediapipe Landmarks to recognizable Gestures
class HandRecog:
    
    def __init__(self, hand_label):
        self.finger = 0
        self.ori_gesture = Gest.PALM
        self.prev_gesture = Gest.PALM
        self.frame_count = 0
        self.hand_result = None
        self.hand_label = hand_label
    
    def update_hand_result(self, hand_result):
        self.hand_result = hand_result

    def get_signed_dist(self, point):
        sign = -1
        if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y:
            sign = 1
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist * sign
    
    def get_dist(self, point):
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist
    
    def get_dz(self, point):
        return abs(self.hand_result.landmark[point[0]].z - self.hand_result.landmark[point[1]].z)
    
    def set_finger_state(self):
        if self.hand_result is None:
            return

        points = [[8, 5, 0], [12, 9, 0], [16, 13, 0], [20, 17, 0]]
        self.finger = 0
        self.finger = self.finger | 0  # thumb
        for idx, point in enumerate(points):
            dist = self.get_signed_dist(point[:2])
            dist2 = self.get_signed_dist(point[1:])
            try:
                ratio = round(dist / dist2, 1)
            except ZeroDivisionError:  # Handle division by zero
                ratio = round(dist / 0.01, 1)

            self.finger = self.finger << 1
            if ratio > 0.5:
                self.finger = self.finger | 1

    def get_gesture(self):
        if self.hand_result is None:
            return Gest.PALM

        current_gesture = Gest.PALM
        if self.finger in [Gest.LAST3, Gest.LAST4] and self.get_dist([8, 4]) < 0.05:
            if self.hand_label == HLabel.MINOR:
                current_gesture = Gest.PINCH_MINOR
            else:
                current_gesture = Gest.PINCH_MAJOR

        elif Gest.FIRST2 == self.finger:
            point = [[8, 12], [5, 9]]
            dist1 = self.get_dist(point[0])
            dist2 = self.get_dist(point[1])
            ratio = dist1 / dist2
            if ratio > 1.7:
                current_gesture = Gest.V_GEST
            else:
                if self.get_dz([8, 12]) < 0.1:
                    current_gesture = Gest.TWO_FINGER_CLOSED
                else:
                    current_gesture = Gest.MID
            
        else:
            current_gesture = self.finger
        
        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0

        self.prev_gesture = current_gesture

        if self.frame_count > 4:
            self.ori_gesture = current_gesture
        return self.ori_gesture

# Executes commands according to detected gestures
class Controller:
    tx_old = 0
    ty_old = 0
    trial = True
    flag = False
    grabflag = False
    pinchmajorflag = False
    pinchminorflag = False
    pinchstartxcoord = None
    pinchstartycoord = None
    pinchdirectionflag = None
    prevpinchlv = 0
    pinchlv = 0
    framecount = 0
    prev_hand = None
    pinch_threshold = 0.3
    
    @staticmethod
    def getpinchylv(hand_result):
        dist = round((Controller.pinchstartycoord - hand_result.landmark[8].y) * 10, 1)
        return dist

    @staticmethod
    def getpinchxlv(hand_result):
        dist = round((hand_result.landmark[8].x - Controller.pinchstartxcoord) * 10, 1)
        return dist
    
    @staticmethod
    def changesystembrightness():
        currentBrightnessLv = sbcontrol.get_brightness() / 100.0
        currentBrightnessLv += Controller.pinchlv / 50.0
        if currentBrightnessLv > 1.0:
            currentBrightnessLv = 1.0
        elif currentBrightnessLv < 0.0:
            currentBrightnessLv = 0.0       
        sbcontrol.fade_brightness(int(100 * currentBrightnessLv), start=sbcontrol.get_brightness())
    
    @staticmethod
    def changesystemvolume():
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        currentVolumeLv = volume.GetMasterVolumeLevelScalar()
        currentVolumeLv += Controller.pinchlv / 50.0
        if currentVolumeLv > 1.0:
            currentVolumeLv = 1.0
        elif currentVolumeLv < 0.0:
            currentVolumeLv = 0.0
        volume.SetMasterVolumeLevelScalar(currentVolumeLv, None)
    
    @staticmethod
    def scrollVertical():
        pyautogui.scroll(120 if Controller.pinchlv > 0.0 else -120)
        
    @staticmethod
    def scrollHorizontal():
        pyautogui.keyDown('shift')
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(-120 if Controller.pinchlv > 0.0 else 120)
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('shift')

    @staticmethod
    def get_position(hand_result):
        point = 9
        position = [hand_result.landmark[point].x, hand_result.landmark[point].y]
        sx, sy = pyautogui.size()
        x_old, y_old = pyautogui.position()
        x = int(position[0] * sx)
        y = int(position[1] * sy)
        if Controller.prev_hand is None:
            Controller.prev_hand = x, y
        delta_x = x - Controller.prev_hand[0]
        delta_y = y - Controller.prev_hand[1]
        if abs(delta_x) > 5 or abs(delta_y) > 5:
            pyautogui.moveTo(x, y)
            Controller.prev_hand = x, y

    @staticmethod
    def handle_controls(gesture, hand_result):
        if gesture == Gest.PALM and Controller.trial:
            Controller.trial = False
            Controller.flag = not Controller.flag
        
        if Controller.flag:
            if gesture == Gest.PINCH_MAJOR and not Controller.pinchmajorflag:
                Controller.pinchmajorflag = True
                Controller.pinchstartxcoord = hand_result.landmark[8].x
                Controller.pinchstartycoord = hand_result.landmark[8].y
            elif gesture == Gest.PINCH_MINOR and not Controller.pinchminorflag:
                Controller.pinchminorflag = True
                Controller.pinchstartxcoord = hand_result.landmark[8].x
                Controller.pinchstartycoord = hand_result.landmark[8].y

            if gesture == Gest.PINCH_MAJOR and Controller.pinchmajorflag:
                Controller.pinchlv = Controller.getpinchylv(hand_result)
                Controller.changesystembrightness()
            elif gesture == Gest.PINCH_MINOR and Controller.pinchminorflag:
                Controller.pinchlv = Controller.getpinchxlv(hand_result)
                Controller.changesystemvolume()

        if gesture == Gest.PALM and Controller.pinchmajorflag:
            Controller.pinchmajorflag = False
        elif gesture == Gest.PALM and Controller.pinchminorflag:
            Controller.pinchminorflag = False
    
    @staticmethod
    def main():
        cap = cv2.VideoCapture(0)
        with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Ignoring empty camera frame.")
                    continue

                # Flip the image horizontally for a later selfie-view display
                frame = cv2.flip(frame, 1)
                # Convert the BGR image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                hand_result = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if hand_result.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness):
                        gesture_recog = HandRecog(HLabel.MAJOR if handedness.classification[0].label == 'Left' else HLabel.MINOR)
                        gesture_recog.update_hand_result(hand_landmarks)
                        gesture_recog.set_finger_state()
                        gesture = gesture_recog.get_gesture()

                        # Display the hand annotations on the image
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Handle gesture controls
                        Controller.get_position(hand_landmarks)
                        Controller.handle_controls(gesture, hand_landmarks)

                # Show the image with the gesture annotations
                cv2.imshow('Hand Gesture Recognition', image)

                # Exit the loop if the user presses 'q'
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

# Entry Point
if __name__ == "__main__":
    Controller.main()
