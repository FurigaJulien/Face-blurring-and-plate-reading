import streamlit as st
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import torch
import pafy
import time
import os
import pandas as pd
import pytesseract


st.sidebar.image('images/logo.png')



col1,col2,col3 = st.columns(3)

with col1:
    nb_of_number_on_plate = st.select_slider("Number of character on plate",[i for i in range(3,10)])
with col2:
    plate_color = st.selectbox("Plate color",['Light','Dark'])


selectbox = st.selectbox("Select an input video:",["Youtube","Camera","Upload a file"])


if selectbox == "Youtube":
    st.text(' Test url : https://www.youtube.com/watch?v=oJ1sAD7IoNs')
    input = st.text_input('Inserez une URL Youtube')
if selectbox == 'Upload a file':
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # To read file as bytes:
        with open(os.path.join("fileDir",uploaded_file.name),"wb") as f:
            f.write((uploaded_file).getbuffer())
    file_availables_list = []
    for filepath in os.listdir("fileDir"):
        file_availables_list.append(filepath)
    if len(file_availables_list) >= 1:
        file_selector  = st.selectbox('Choose an already existing file',file_availables_list)

col1,col2,col3 = st.columns(3)

with col1:
    start_button = st.button("Start inference",key='start_button')
with col2:
    stop_button = st.button("Stop inference",key='stop_button')


@st.cache(allow_output_mutation=True)
def get_cap(value):
    return cv2.VideoCapture(value)

if start_button:

    predicted_license_plates = set()
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_nano.pt')

    if selectbox == "Youtube":
        video = pafy.new(input)
        best = video.getbestvideo(preftype='mp4')
        output = best.url

    if selectbox == "Camera":
        output = 0

    if selectbox == "Upload a file":
        output = "fileDir/" + file_selector

    capture = get_cap(output)
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
                    sub_licence = cv2.resize(sub_licence, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                    gray = cv2.cvtColor(sub_licence, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray,(3,3),0)
                    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
                    img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
                    if plate_color == 'Dark':
                        img = cv2.bitwise_not(img)
                    predicted_result = pytesseract.image_to_string(img, lang ='eng',config ='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

                    
                    filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace(' ','')
                    print(len(filter_predicted_result))
                    if len(filter_predicted_result)==int(nb_of_number_on_plate):
                        predictions.append(filter_predicted_result)
                        if predictions.count(filter_predicted_result)>5:
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
        t.table(pd.DataFrame({'Plates detected':list(predicted_license_plates)}))

        if stop_button:
            capture.clear()


    
