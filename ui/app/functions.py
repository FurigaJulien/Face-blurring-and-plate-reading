import cv2
import numpy as np
import streamlit as st
import pandas as pd
from Database import Users

def preprocess_frames(frame:np.ndarray,y1:int,y2:int,x1:int,x2:int,plate_color:str)->np.ndarray:
    """_summary_

    Parameters
    ----------
    frame : np.ndarray
        input image
    y1 : int
        position of the top left corner of the plate
    y2 : int
        position of the bottom right corner of the plate
    x1 : int
        position of the top left corner of the plate
    x2 : int
        position of the bottom right corner of the plate
    plate_color : str
        plate color

    Returns
    -------
    np.ndarray
        returns the cropped plate
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
    """converts a dataframe to a numpy array

    Parameters
    ----------
    df : pd.DataFrame
        original dataframe

    Returns
    -------
    _type_
        returns csv file
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
    
    if st.session_state["password_correct"] == True:
        return True


    if not 'signin' in st.session_state.keys():
        st.session_state["signin"] = False


    def password_entered():
        """Checks whether a password entered by the user is correct."""
        valid,user_id = Users.check_password(engine,st.session_state["username"],st.session_state["password"])
        if valid:
            st.session_state["password_correct"] = True
            st.session_state['user_id'] = user_id
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]

        else:
            st.session_state["password_correct"] = False


    if not st.session_state['signin']:
        with st.form(key='my_form',clear_on_submit=True):
            username_placeholder = st.text_input("Username", key="username")
            password_placeholder = st.text_input("Password", type="password", key="password")
            col1,col2,col3 = st.columns(3)

            with col1:
                check_button = st.form_submit_button("Connect")
            with col2:
                signin_button = st.form_submit_button("Signin")

        if check_button:

            password_entered()
            if not st.session_state["password_correct"]:
                st.error("😕 User not known or password incorrect")
                return False
            else:

                return True
        
        if signin_button:
            
            st.session_state["signin"] = True
            check_password(engine)


    else:
        with st.form(key='my_form_signin'):
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

            check_button = button_placeholder.form_submit_button("Create account")
            login_button = login_button_placeholder.form_submit_button('Return to login')


        if check_button :

            if Users.get_username_availability(engine,username):
                user = Users(username=username,password=password,first_name=first_name,family_name=family_name)
                Users.insert_user(engine,user)
                username_placeholder.empty()
                password_placeholder.empty()
                first_name_placeholder.empty()
                family_name_placeholder.empty()

                
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

