import streamlit as st
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import torch
import pafy
import time
import pytesseract
import glob
import os


class VideoTransformer(VideoTransformerBase):

    classnames = ['face', 'licence']
    label = {}
    for i, name in enumerate(classnames):
        label[i]=name
    # load pre-trained model


    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    


    def transform(self, frame):

        image = frame.to_ndarray(format="bgr24")
        start = time.time()
        prediction =  self.model(image).pandas().xyxy[0].values
        end=time.time()
        print(end - start)
        if prediction is not None:
            for pred in prediction:

                x1 = int(pred[0])
                y1 = int(pred[1])
                x2 = int(pred[2])
                y2 = int(pred[3])

                start = (x1,y1)
                end = (x2,y2)

                name = pred[-1]
                color = (0,255,0)
                if name =='licence':
                    sub_plate = image[y1:y2, x1:x2]
                    image = cv2.rectangle(image, start, end, color)
                    image = cv2.putText(image, name, (x1,y1+25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA) 
                    pytesseract.pytesseract.tesseract_cmd = (r"C:\Program Files (x86)\Tesseract-OCR\tesseract")

                    print(pytesseract.image_to_string(sub_plate))

                else:
                    sub_face = image[y1:y2, x1:x2]
                    # apply a gaussian blur on this new recangle image
                    temp = cv2.resize(sub_face, (5, 5), interpolation=cv2.INTER_LINEAR)
                    sub_face = cv2.resize(temp, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
                    # merge this blurry rectangle to our final image
                    image[y1:y1+sub_face.shape[0], x1:x1+sub_face.shape[1]] = sub_face
        
        return image
       



    


webrtc_streamer(key="example",video_transformer_factory=VideoTransformer)
st.text(' Test url : https://www.youtube.com/watch?v=oJ1sAD7IoNs')
url = st.text_input('Inserez une URL Youtube')


if len(url)>3:
    predicted_license_plates = set()
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    video = pafy.new(url)
    best = video.getbestvideo(preftype='mp4')

    capture = cv2.VideoCapture(best.url)
    stframe = st.empty()

    predictions = []
    t = st.empty()
    i=0
    while capture.isOpened():
        i+=1
        ret, frame = capture.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        prediction =  model(image).pandas().xyxy[0].values
            

        if prediction is not None:
            for pred in prediction:

                x1 = int(pred[0])
                y1 = int(pred[1])
                x2 = int(pred[2])
                y2 = int(pred[3])

                start = (x1,y1)
                end = (x2,y2)

                name = pred[-1]
                color = (0,255,0)
                
                if name =='licence':
                    
                    sub_licence = image.copy()[y1:y2, x1:x2]
                    
                    image = cv2.rectangle(image, start, end, color)
                    image = cv2.putText(image, name, (x1,y1+25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA) 
# the second one 
                    sub_licence = cv2.resize(sub_licence, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                    gray = cv2.cvtColor(sub_licence, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray,(3,3),0)
                    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
                    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
                    img_not = cv2.bitwise_not(opening)
                    predicted_result = pytesseract.image_to_string(img_not, lang ='eng',config ='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

                    
                    filter_predicted_result = "".join(predicted_result.split()).replace(":", "")

                    if len(filter_predicted_result)>4:
                        predictions.append(filter_predicted_result)
                        if predictions.count(filter_predicted_result)>9:
                            predicted_license_plates.add(filter_predicted_result)
                            predictions = []
                            

                else:
                    sub_face = image[y1:y2, x1:x2]
                    # apply a gaussian blur on this new recangle image
                    temp = cv2.resize(sub_face, (5, 5), interpolation=cv2.INTER_LINEAR)
                    sub_face = cv2.resize(temp, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
                    # merge this blurry rectangle to our final image
                    image[y1:y1+sub_face.shape[0], x1:x1+sub_face.shape[1]] = sub_face

        stframe.image(image)
        t.write('plates: ' + str(predicted_license_plates))




    
