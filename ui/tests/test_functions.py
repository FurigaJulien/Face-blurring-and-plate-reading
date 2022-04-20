import pytest
import numpy as np
from app.functions import preprocess_frames

@pytest.mark.parametrize("frame,y1,y2,x1,x2,plate_color,expected", [(np.zeros((100,100,3),dtype=np.uint8),0,100,0,100,'Dark',(300,300),),
(np.zeros((100,100,3),dtype=np.uint8),0,100,0,100,'Light',(300,300),),(np.zeros((300,300,3),dtype=np.uint8),0,10,0,10,'Dark',(30,30),),])
def test_preprocess_frames(frame:np.ndarray,y1:int,y2:int,x1:int,x2:int,plate_color:str,expected:tuple):
    """Test preprocess function

    Parameters
    ----------
    frame : np.ndarray
        original frame
    y1 : int
        position of the top left corner of the licence plate
    y2 : int
        position of the bottom right corner of the licence plate
    x1 : int
        position of the top left corner of the licence plate
    x2 : int
        position of the bottom right corner of the licence plate
    plate_color : str
        dominant plate color
    expected : tuple
        size of the final frame
    """
    sub_frame = preprocess_frames(frame,y1,y2,x1,x2,plate_color)

    assert sub_frame.shape == expected










