# IMPORT STATEMENTS
import cv2 
import mediapipe as mp
import argparse

#ARGUMENT PARSER 
args = argparse.ArgumentParser()
args.add_argument('--mode', default='webcam')
args.add_argument('--filePath', default=None)
args = args.parse_args()

#IMPORT FUNCTION FOR DETECTING FACE
from util import detect_face

#INITIALIZE FACE DETECTION OBJECT
mp_face_detection = mp.solutions.face_detection

#MAIN FUNCTION
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    if args.mode in ['image']:
        img = cv2.imread(args.filePath)
        img = cv2.resize(img , (450 , 500))
        img = detect_face(img=img , face_detection_object=face_detection)

        cv2.imshow('img' , img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    elif args.mode in ['video']:
        cap = cv2.VideoCapture(args.filePath)
        ret , frame = cap.read()
        while ret: 
            frame = detect_face(img=frame , face_detection_object=face_detection)
            cv2.imshow('img' , frame)
            if cv2.waitKey(40) & 0xFF == ord('q'):
                break

            ret  ,frame = cap.read()
        
        cap.release()


    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)
        ret , frame = cap.read()
        while ret: 
            frame = detect_face(img=frame , face_detection_object=face_detection , W=frame.shape[1] , H=frame.shape[0])
            cv2.imshow('img' , frame)
            if cv2.waitKey(40) & 0xFF == ord('q'):
                break

            ret , frame = cap.read()
                
        cap.release()

    