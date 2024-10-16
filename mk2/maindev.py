# Main Program - Face Analysis Dev Tool Drowsiness Detection Dev
# Karamelsoft Inc 2024
# Install 'cv2', 'streamlit' library by 'pip3 install opencv-python' and 'pip3 install streamlit'
# Open a new Terminal, type 'streamlit run fadevmain.py' to run this main program

# main ui with detection function

import cv2
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import time
import os
import dlib
import imutils
from imutils import face_utils
import timeit
import make_train_data as mtd
from scipy.spatial import distance as dist
from threading import Thread
from threading import Timer
import ringing_alarm as alarm

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
OPEN_EAR = 0 #For init_open_ear()
EAR_THRESH = 0 #Threashold value
TIMER_FLAG = False #Flag to activate 'start_closing' variable, which measures the eyes closing time.
ALARM_FLAG = False #Flag to check if alarm has ever been triggered.
EAR_CONSEC_FRAMES = 20
COUNTER = 0 #Frames counter.
test_data = []
result_data = []
closed_eyes_time = [] #The time eyes were being offed.

power, nomal, short = mtd.start(25)


# Main Page title, icon
st.set_page_config(page_title='Face Analysis Dev Tool',
                   page_icon='karamellogo.ico',)

# Main Page left side bar(menu)
with st.sidebar:
  st.title('Drowsiness Detection Dev')
  st.write('---')
  st.write('## Step 1. Detection')
  st.selectbox('Choose a Detection Model', ('Dlib', 'AI'))
  uploaded_file_vid = st.file_uploader('Choose a Video File... [mp4/mov/avi/wmv]')
  cam_id = st.selectbox('Or Choose a Live Camera', ('None', 'CAM 0','CAM 1'))  
  st.write('---')
  st.write('## Step 2. Verification')
  uploaded_file_pjt = st.file_uploader('Choose a Project File')
  st.write('---')
  st.write('## Step 3. Evaluation')
  uploaded_file_rpt = st.file_uploader('Choose a Report File')
  st.write('---')
  st.write('Copyright Karamelsoft Inc. 2024')

# Main Page center
main_tab, setting_tab, report_tab = st.tabs(["Main", "Settings", "Report"])
frame_placeholder = st.empty()
chart_placeholder = st.empty()
columns_placeholder = st.empty()
ear_data = []
ear_value = 0
frame_order = 0
vid_file = 'None'
if ear_data not in st.session_state:
  st.session_state.chart_data = ear_data

folder_name = "processed"
if not os.path.exists(folder_name):
  os.mkdir(folder_name)

def eye_aspect_ratio(eye) :
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def light_removing(frame) :
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L = lab[:,:,0]
    med_L = cv2.medianBlur(L,99) #median filter
    invert_L = cv2.bitwise_not(med_L) #invert lightness
    composed = cv2.addWeighted(gray, 0.75, invert_L, 0.25, 0)
    return L, composed

def get_ear_value(img, frame_order):  

# Original AWAKE Detection module
  both_ear = 0
  L, gray = light_removing(img)
  rects = detector(gray, 0)
  for rect in rects:
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    # (leftEAR + rightEAR) / 2 => both_ear.
    both_ear = (leftEAR + rightEAR) * 500  # I multiplied by 1000 to enlarge the scope.

    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

    if both_ear < EAR_THRESH:
        if not TIMER_FLAG:
            start_closing = timeit.default_timer()
            TIMER_FLAG = True
        COUNTER += 1

        if COUNTER >= EAR_CONSEC_FRAMES:

            mid_closing = timeit.default_timer()
            closing_time = round((mid_closing - start_closing), 3)

            if closing_time >= RUNNING_TIME:
                if RUNNING_TIME == 0:
                    CUR_TERM = timeit.default_timer()
                    OPENED_EYES_TIME = round((CUR_TERM - PREV_TERM), 3)
                    PREV_TERM = CUR_TERM
                    RUNNING_TIME = 1.75

                RUNNING_TIME += 2
                ALARM_FLAG = True
                ALARM_COUNT += 1

                print("{0}st ALARM".format(ALARM_COUNT))
                print("The time eyes is being opened before the alarm went off :", OPENED_EYES_TIME)
                print("closing time :", closing_time)
                test_data.append([OPENED_EYES_TIME, round(closing_time * 10, 3)])
                result = mtd.run([OPENED_EYES_TIME, closing_time * 10], power, nomal, short)
                result_data.append(result)
                t = Thread(target=alarm.select_alarm, args=(result,))
                t.deamon = True
                t.start()
    else:
        COUNTER = 0
        TIMER_FLAG = False
        RUNNING_TIME = 0
        ALARM_FLAG = False

        if ALARM_FLAG:
            end_closing = timeit.default_timer()
            closed_eyes_time.append(round((end_closing - start_closing), 3))
            print("The time eyes were being offed :", closed_eyes_time)   

    cv2.putText(img, "EAR : {:.2f}".format(both_ear), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
  file_name = os.path.join(folder_name, f'{frame_order}.png')
  cv2.imwrite(file_name, img)  
  return both_ear, img

with main_tab:    
  save_result = st.button('Save Result', key='1')
  if uploaded_file_vid is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
      temp_file.write(uploaded_file_vid.read())
      vid_file = temp_file.name    
  elif cam_id == 'CAM 0' or cam_id == 'CAM 1':
    vid_file = 0  
  if vid_file != 'None':
    cap = cv2.VideoCapture(vid_file)
    if cap.isOpened():
      fps=cap.get(cv2.CAP_PROP_FPS)    
      while True:
        ret, img = cap.read()        
        if ret:
          img = imutils.resize(img, width=400)
          frame_order += 1
          ear_value, img = get_ear_value(img, frame_order)
          frame_placeholder.image(img, channels='BGR')       
          st.session_state.chart_data.append(ear_value)        
          chart_data = pd.Series(st.session_state.chart_data)
          chart_placeholder.line_chart(chart_data, height=200, x_label='frame', y_label="EAR")
          col1, col2 = columns_placeholder.columns(2)      
          col1.metric(label = 'EAR', value = int(ear_value))
          col2.metric(label = 'frame', value = frame_order )
          cv2.waitKey(1)   
        else:
          break      
    else:
      print("Can't open video")
    cap.release()  

with setting_tab:
  col1, col2 = st.columns(2) 
  with col1:
    st.slider(label='EAR Warning Level', max_value=300, min_value=0)
    st.slider(label='EAR Danger Level', max_value=300, min_value=0)
  with col2:
    st.number_input(label='Eye Blink Time', max_value=1, min_value=0)
with report_tab:
  st.subheader('Missing Frame Count')
  col1, col2, col3 = st.columns(3)
  col1.metric('Total Frames', 700)
  col2.metric('Missing Frames', 5)
  col3.metric('Detection Score', '99.95%')
  st.subheader('Final Accuracy')
  col1, col2, col3 = st.columns(3)
  col1.metric('Total Frames', 700)
  col2.metric('Missing Frames', 5)
  col3.metric('Detection Score', '99.95%')