import cv2 as cv
import glob
import numpy as np
import sys
from scipy import linalg
import yaml
import os


#This will contain the calibration settings from the calibration_settings.yaml file
calibration_settings = {}


def parse_calibration_settings_file(filename):
    
    global calibration_settings

    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()
    
    print('Using for calibration settings: ', filename)

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    #rudimentray check to make sure correct file was loaded
    if 'camera0' not in calibration_settings.keys():
        print('camera0 key was not found in the settings file. Check if correct calibration_settings.yaml file was passed')
        quit()

#Open camera stream and save frames
def save_frames_single_camera(camera_name):

    #create frames directory
    if not os.path.exists('frames'):
        os.mkdir('frames')

    #get settings
    camera_device_id = calibration_settings[camera_name]
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    number_to_save = calibration_settings['mono_calibration_frames']
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']

    #open video stream and change resolution.
    #Note: if unsupported resolution is used, this does NOT raise an error.
    cap = cv.VideoCapture(camera_device_id)
    cap.set(3, width)
    cap.set(4, height)
    
    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
    
        ret, frame = cap.read()
        if ret == False:
            #if no video data is received, can't calibrate the camera, so exit.
            print("No video data received from camera. Exiting...")
            quit()

        frame_small = cv.resize(frame, None, fx = 1/view_resize, fy=1/view_resize)

        if not start:
            cv.putText(frame_small, "Press SPACEBAR to start collection frames", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        
        if start:
            cooldown -= 1
            cv.putText(frame_small, "Cooldown: " + str(cooldown), (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv.putText(frame_small, "Num frames: " + str(saved_count), (50,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            
            #save the frame when cooldown reaches 0.
            if cooldown <= 0:
                savename = os.path.join('frames', camera_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame)
                saved_count += 1
                cooldown = cooldown_time

        cv.imshow('frame_small', frame_small)
        k = cv.waitKey(1)
        
        if k == 27:
            #if ESC is pressed at any time, the program will exit.
            quit()

        if k == 32:
            #Press spacebar to start data collection
            start = True

        #break out of the loop when enough number of frames have been saved
        if saved_count == number_to_save: break

    cv.destroyAllWindows()


if __name__ == '__main__':

    #Open and parse the settings file
    parse_calibration_settings_file(sys.argv[1])

    """Step1. Save calibration frames for single cameras"""
    save_frames_single_camera('camera0') #save frames for camera0
    save_frames_single_camera('camera1') #save frames for camera1