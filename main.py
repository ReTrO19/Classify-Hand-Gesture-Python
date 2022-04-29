import cv2
from hand_details import HandEncodings
import pickle
import mediapipe as mp
import numpy
from utils import feature_extract
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture("test.mp4")
# Loading Classes
with open("classes.txt",'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
# classes = ["Neutral","Nice","Peace"]
# frame = cv2.imread("hand.jpg")

video_writer = cv2.VideoWriter("OutVideo.mp4", 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (640,480))
while True:

    ret,frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame,(640,480))
    # Detecting Hands from the frame
    with mp_hands.Hands(static_image_mode = False,max_num_hands = 1,
        min_detection_confidence = 0.5,model_complexity = 0) as hands:
        predicted_class = None
        HEclass = HandEncodings(frame,hands,mp_hands)
        TF,IF,MF,RF,PF = HEclass.frame_to_encodings()
        if TF is not None:
            TF_IF,IF_MF,MF_RF,RF_PF = feature_extract(TF,IF,MF,RF,PF)
            feature_vector = numpy.array([[TF_IF,IF_MF,MF_RF,RF_PF]])
            loaded_model = pickle.load(open('knnweight_file', 'rb'))
            result = loaded_model.predict(feature_vector)
            predicted_class = classes[result[0]]
        
        frame = cv2.putText(frame, 'Hand Gesture:'+str(predicted_class), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    video_writer.write(frame)
    cv2.imshow("Output",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   

cap.release()
video_writer.release()

cv2.destroyAllWindows()