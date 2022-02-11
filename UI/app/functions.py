import cv2
import numpy as np
from pylibsrtp import Session
import streamlit as st
import pandas as pd
from Database import Users

def preprocess_frames(frame:np.ndarray,y1:int,y2:int,x1:int,x2:int,plate_color:str)->np.ndarray:
    """
    Functions to preprocess frames for pytesseract
    """
    sub_licence = frame[y1:y2, x1:x2]
    sub_licence = cv2.resize(sub_licence, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(sub_licence, cv2.COLOR_BGR2GRAY)

    if plate_color == 'Dark':
        invert = 255 - gray
        return invert
    else:
        return gray

@st.cache
def convert_df(df:pd.DataFrame):
    """
    Convert dataframe to csv file
    IMPORTANT: Cache the conversion to prevent computation on every rerun
    """
    return df.to_csv().encode('utf-8')


def get_cap(value:str)->None:
    """
    Create a cv2.VideoCapture object kept in cache to give streamlit the ability to call the release function"""
    if 'capture' in st.session_state.keys():
        st.session_state['capture'].release()
    st.session_state['capture'] = cv2.VideoCapture(value)


def check_password(engine):
    """Returns `True` if the user had a correct password."""


    if not 'password_correct' in st.session_state.keys():
        st.session_state["password_correct"] = False
    


    if not 'signin' in st.session_state.keys():
        st.session_state["signin"] = False


    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if Users.check_password(engine,st.session_state["username"],st.session_state["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]

        else:
            st.session_state["password_correct"] = False


    if not st.session_state['signin']:
        username_placeholder = st.empty()
        password_placeholder = st.empty()
        col1,col2,col3 = st.columns(3)
        with col1:
            button_placeholder = st.empty()
        with col2:
            signin_button_placeholder = st.empty()

        username_placeholder.text_input("Username", key="username")
        password_placeholder.text_input("Password", type="password", key="password")

        check_button = button_placeholder.button("Connect")
        signin_button = signin_button_placeholder.button("Signin")
        
        if check_button:
            password_entered()
            if not st.session_state["password_correct"]:
                st.error("😕 User not known or password incorrect")
                return False
            else:
                username_placeholder.empty()
                password_placeholder.empty()
                button_placeholder.empty()
                signin_button_placeholder.empty()
                return True
        
        if signin_button:
            
            username_placeholder.empty()
            password_placeholder.empty()
            button_placeholder.empty()
            signin_button_placeholder.empty()
            st.session_state["signin"] = True
            check_password(engine)


    else:
        col1,col2 = st.columns(2)

        with col1:
            username_placeholder = st.empty()
            first_name_placeholder = st.empty()
        
        with col2:
            password_placeholder = st.empty()
            family_name_placeholder = st.empty()

        col1,col2,col3 = st.columns(3)
        with col1:
            button_placeholder = st.empty()
        with col2:
            login_button_placeholder = st.empty()


        username = username_placeholder.text_input("Username",on_change=None)
        password = password_placeholder.text_input("Password", type="password")
        first_name = first_name_placeholder.text_input("Insert your name")
        family_name = family_name_placeholder.text_input("Insert your family name")

        check_button = button_placeholder.button("Create account")
        login_button = login_button_placeholder.button('Return to login')


        if check_button:

            if Users.get_username_availability(engine,username):
                user = Users(username=username,password=password,first_name=first_name,family_name=family_name)
                Users.insert_user(engine,user)
                username_placeholder.empty()
                password_placeholder.empty()
                first_name_placeholder.empty()
                family_name_placeholder.empty()

                check_button = button_placeholder.empty()
                login_button_placeholder.empty()
                st.session_state['signin'] = False
                check_password(engine)

            else:
                st.error("This username already exists, try another !")

        if login_button:
            
            username_placeholder.empty()
            password_placeholder.empty()
            first_name_placeholder.empty()
            family_name_placeholder.empty()

            check_button = button_placeholder.empty()
            login_button_placeholder.empty()
            st.session_state['signin'] = False
            check_password(engine)

