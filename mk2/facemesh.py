# Private library - Head Pose Estimation by MediaPipe
# Karamelsoft Inc 2024
# Inspired by Nicolai Nielson youtube https://www.youtube.com/watch?v=-toNMaS4SeQ&t=475s
# pip install mediapipe

# import public library
import cv2
import mediapipe as mp
import numpy as np
import time

# Head Pose Estimation module
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def get_face_mesh(img):
    start = time.perf_counter()
    # image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results=face_mesh.process(image)
    image.flags.writeable=True
    image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_h, img_w, img_c=image.shape
    face_3d=[]
    face_2d=[]
    fps =  1

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx==263 or idx==1 or idx==61 or idx==291 or idx==199:
                    if idx==1:
                        nose_2d=(lm.x*img_w, lm.y*img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z*3000)
                    x,y=int(lm.x*img_w), int(lm.y*img_h)
                    face_2d.append([x,y])
                    face_3d.append([x, y, lm.z])
            face_2d=np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            focal_length=1*img_w
            cam_matrix=np.array([
                [focal_length, 0, img_h/2],[0,focal_length, img_w/2],[0,0,1]
            ])
            dist_matrix=np.zeros((4,1),dtype=np.float64)
            success, rot_vec, trans_vec=cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac= cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx,Qy,Qz=cv2.RQDecomp3x3(rmat)
            x=angles[0]*360 # Pitching, 위보기는 +,  아래 보기는 -
            y=angles[1]*360 # Yawing, 본인 기준 오른쪽 보기 +, 왼쪽 보기 -
            z=angles[2]*360 # Rolling, 본인 기준 오른쪽으로 머리 기울기 +, 왼쪽으로 머리 기울기 -

            if y<-10:
                text="looking left"
            elif y>10:
                text='looing right'
            elif x<-5:
                text='looking down'
            elif x>5:
                text='looking up'
            else:
                text='forward'

            nose_3d_projection, jacobian=cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1=(int(nose_2d[0]), int(nose_2d[1]))
            p2=(int(nose_2d[0]+y*10), int(nose_2d[1]-x*10))

            cv2.line(image, p1, p2, (255,0,0),3)

            cv2.putText(image, text, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0),2)
            cv2.putText(image, "x: "+str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255), 2)
            cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            end=time.perf_counter()
            totalTime=end-start
            if not totalTime==0:
                fps=1/totalTime
            else:
                totalTime=1
            print("FPS: ", fps)

            cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,255,0),2)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,

                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )
    return image
        
if __name__ == "__main__":
  print("This is not a main program.")
