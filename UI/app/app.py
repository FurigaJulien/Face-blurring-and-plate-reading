import streamlit as st
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import torch
import pafy
from functions import preprocess_frames,check_password,convert_df,get_cap
import os
import easyocr
import pandas as pd
from dotenv import load_dotenv
from sqlmodel import create_engine
import os

load_dotenv("env/streamlit.env")
db_user = os.getenv('POSTGRES_USER')
db_password = os.getenv("POSTGRES_PASSWORD")
db_name = os.getenv("POSTGRES_DB")
db_address = os.getenv("POSTGRES_ADDRESS")
db_port = os.getenv("POSTGRES_PORT")
postgres_url = f"postgresql://{db_user}:{db_password}@{db_address}:{db_port}/pco"

# Create objects

st.session_state['engine'] = create_engine(postgres_url, echo=True)
st.session_state['reader'] = easyocr.Reader(['en'])
st.sidebar.image('./images/logo.png')

if check_password(st.session_state['engine']):


    # Define session state variable
    if 'plates' not in st.session_state.keys():
        st.session_state['plates'] = set()


    # Sidebar image
    

    # Create columns to display parameters buttons
    col1,col2,col3 = st.columns(3)
    with col1:
        nb_of_number_on_plate = st.select_slider("Number of character on plate",[i for i in range(3,10)])
    with col2:
        plate_color = st.selectbox("Plate color",['Light','Dark'])



    # Create a selectbox to get input type for videoCapture
    selectbox = st.selectbox("Select an input video:",["Camera","Youtube","Upload a file"])
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

    # Create columns for start:stop buttons
    col1,col2,col3 = st.columns(3)
    with col1:
        start_button = st.button("Start inference",key='start_button')
    with col2:
        stop_button = st.button("Stop inference",key='stop_button')



    # Init placeholders to display video and read plates data
    stframe = st.empty()
    plates_placeholder = st.empty()

    # Display data if one plate or more has been detected
    if len(st.session_state['plates'])>0:
        df = pd.DataFrame({'Plates detected':list(st.session_state['plates'])})
        csv = convert_df(df)
        plates_placeholder.table(df)

        st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='data.csv',
        mime='text/csv',)

    # Start inference for choosen stream
    if start_button:
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

        get_cap(output)
        predictions = []

        while st.session_state['capture'].isOpened():
            ret, frame = st.session_state['capture'].read()
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
                        

                        img = preprocess_frames(frame,y1,y2,x1,x2,plate_color)
                        image = cv2.rectangle(image, start, end, color)
                        image = cv2.putText(image, name, (x1,y1+25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA) 

                        predicted_result = st.session_state['reader'].readtext(img)
                        try:
                            plate_read = ''
                            for val in predicted_result:
                                plate_read += val[1]
                            print(plate_read)
                            filter_predicted_result = "".join(plate_read.split()).replace(":", "").replace(' ','').replace('-','')
                            if len(filter_predicted_result)==int(nb_of_number_on_plate):
                                predictions.append(filter_predicted_result)
                                if predictions.count(filter_predicted_result)>5:
                                    st.session_state['plates'].add(filter_predicted_result)
                                    predictions = []
                        except:
                            pass

                    else:
                        sub_face = image[y1:y2, x1:x2]
                        temp = cv2.resize(sub_face, (5, 5), interpolation=cv2.INTER_LINEAR)
                        sub_face = cv2.resize(temp, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
                        image[y1:y1+sub_face.shape[0], x1:x1+sub_face.shape[1]] = sub_face


            stframe.image(image)
            plates_placeholder.table(pd.DataFrame({'Plates detected':list(st.session_state['plates'])}))


            if stop_button:
                if 'capture' in st.session_state.keys():
                    st.session_state['capture'].release()



        
