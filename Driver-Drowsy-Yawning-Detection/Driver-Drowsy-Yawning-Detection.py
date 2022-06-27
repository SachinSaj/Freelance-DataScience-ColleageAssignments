from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
from playsound import playsound

EYES_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.68
frame_check = 20 
frame_check_mouth =15
detect = dlib.get_frontal_face_detector() # To detect Face
predict = dlib.shape_predictor("Model/shape_predictor_68_face_landmarks.dat") # To detect 68 Facial Features

def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = distance.euclidean(mouth[2], mouth[10])  # 51, 59
    B = distance.euclidean(mouth[4], mouth[8])  # 53, 57
    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = distance.euclidean(mouth[0], mouth[6])  # 49, 55

    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    # return the mouth aspect ratio
    return mar

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    
    # compute the eyes aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear

# Getting the Facial landmarks points for both the eyes and mouth 
# Total Facial Landmarks points on face is 68.
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

cap=cv2.VideoCapture(0)
flag=0
counter = 0
while True:
    ret, frame=cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        
        # Calculating for Drowsiness
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        if ear < EYES_AR_THRESH:
            flag += 1
            if flag >= frame_check_mouth:
                playsound('Sound/urgent_sound-2.wav')
        else:
            flag = 0
        
        # Calculating for Yawning
        mouth = shape[mStart:mEnd]
        mouthMAR = mouth_aspect_ratio(mouth)
        mar = mouthMAR
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0,255,0), 1)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if mar >= MOUTH_AR_THRESH:
            counter += 1
            if counter >= frame_c:
                playsound('Sound/urgent_sound-2.wav')
            
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cv2.destroyAllWindows()
        cap.release()
        break
