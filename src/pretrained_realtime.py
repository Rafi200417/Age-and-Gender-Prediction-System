import cv2
import numpy as np
import os
import threading

MODEL_DIR = os.path.join("D:\Age and Gender Identification\models\caffe")
FACE_PROTO = os.path.join("D:\Age and Gender Identification\models\caffe", "deploy.prototxt")
FACE_MODEL = os.path.join("D:\Age and Gender Identification\models\caffe", "res10_300x300_ssd_iter_140000.caffemodel")
AGE_PROTO = os.path.join("D:\Age and Gender Identification\models\caffe", "age_deploy.prototxt")
AGE_MODEL = os.path.join("D:\Age and Gender Identification\models\caffe", "age_net.caffemodel")
GENDER_PROTO = os.path.join("D:\Age and Gender Identification\models\caffe", "gender_deploy.prototxt")
GENDER_MODEL = os.path.join("D:\Age and Gender Identification\models\caffe", "gender_net.caffemodel")

# Load models using readNetFromCaffe instead of readNet
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

def start_capture(cap):
    print("Starting video capture... Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
            
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
            
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                    
                face = frame[startY:endY, startX:endX]
                if face.size == 0:
                    continue
                    
                face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746))
                    
                gender_net.setInput(face_blob)
                gender_preds = gender_net.forward()
                gender = GENDER_LIST[gender_preds[0].argmax()]
                gender_accuracy = gender_preds[0].max() * 100
                    
                age_net.setInput(face_blob)
                age_preds = age_net.forward()
                age = AGE_LIST[age_preds[0].argmax()]
                age_accuracy = age_preds[0].max() * 100
                    
                label = f"{gender} ({gender_accuracy:.2f}%), {age} ({age_accuracy:.2f}%)"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
        cv2.imshow("Real-Time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def detect_age_gender_video():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    start_capture(cap)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_age_gender_video()