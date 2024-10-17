# Private library - EAR detect by Dlib
# Karamelsoft Inc 2024

# import public library
import cv2
import dlib
from imutils import face_utils
import make_train_data as mtd
import timeit
from threading import Thread
from scipy.spatial import distance as dist
import ringing_alarm as alarm

# Original AWAKE Detection module
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

def get_ear_value(img):  
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
  return both_ear, img

if __name__ == "__main__":
  print("This is not a main program.")