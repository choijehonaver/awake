# Main Program - Face Analysis Dev Tool Drowsiness Detection Dev
# Karamelsoft Inc 2024
# Install 'cv2', 'streamlit' library by 'pip3 install opencv-python' and 'pip3 install streamlit'
# Open a new Terminal, type 'streamlit run fadevmain.py' to run this main program

# import public library
import cv2
import streamlit as st
import pandas as pd
import tempfile
import os
import imutils

# import private library
import make_train_data as mtd
import eardetect as ed
import facemesh as fm

# Main Page title, icon
st.set_page_config(page_title='Face Analysis Dev Tool',
                   page_icon='karamellogo.ico',)

# Main Page left side bar(menu)
with st.sidebar:
  st.title('Drowsiness Detection Dev')
  st.write('---')
  st.write('## Step 1. Detection')
  mode = st.selectbox('Choose a Detection Model', ('MediaPipe','Dlib'))
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
          file_name = os.path.join(folder_name, f'{frame_order}.png') 

          if mode == "MediaPipe":
            img = fm.get_face_mesh(img)
            frame_placeholder.image(img, channels='BGR')

          else: # Dlib
            ear_value, img = ed.get_ear_value(img)
            frame_placeholder.image(img, channels='BGR')       
            st.session_state.chart_data.append(ear_value)        
            chart_data = pd.Series(st.session_state.chart_data)
            chart_placeholder.line_chart(chart_data, height=200, x_label='frame', y_label="EAR")
            col1, col2 = columns_placeholder.columns(2)      
            col1.metric(label = 'EAR', value = int(ear_value))
            col2.metric(label = 'frame', value = frame_order )
          cv2.imwrite(file_name, img)  
          if cv2.waitKey(5) & 0xFF == 27:
            break 
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


  



