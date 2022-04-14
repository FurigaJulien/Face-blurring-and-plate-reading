from tkinter import image_names
import requests
import base64
import io
import os
from loguru import logger
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder


def read_image_and_import_into_roboflow(image_path:str,image_name:str,dataset_name:str,api_key:str):
    """Read an image and import it to Roboflow

    Parameters
    ----------
    path : str
        image path
    image_name : str
        image name 
    dataset_name : str
        dataset name
    api_key : str
        Roboflow api_key
    """

    # Read image
    image = Image.open(image_path)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='jpg')

    # Create multipart form data
    multipart_data = MultipartEncoder(
        fields={
            'file': (image_name, image_bytes.getvalue(), 'image/jpg')
        }
    )

    # Send request
    upload_url = f"https://api.roboflow.com/dataset/{dataset_name}/upload?api_key={api_key}"
    requests.post(upload_url, data=multipart_data, headers={'Content-Type': multipart_data.content_type})

def import_image_annotation(annotation_path:str,annotation_name:str,dataset_name:str,api_key:str):
    """_summary_

    Parameters
    ----------
    annotation_path : str
        annotation path
    image_name : str
        image name 
    dataset_name : str
        dataset name
    api_key : str
        Roboflow api_key
    """

    # Read annotation
    annotation = open(annotation_path, 'r').read()
    image_name = "".join(annotation_name.split('.')[:-1])
    logger.info(f"Importing annotation for {image_name}")
    # Send request
    upload_url = f"https://api.roboflow.com/dataset/{dataset_name}/annotate/{image_name}?api_key={api_key}&name={annotation_path}"
    requests.post(upload_url, data=annotation, headers={'Content-Type': 'text/plain'})


def read_image_from(base_path:str,dataset_name:str):
    """Go throught folders and import image to Roboflow

    Parameters
    ----------
    base_path : str
        base path
    dataset_name : str
        dataset name

    """

    for directory in os.listdir(base_path,dataset_name):
        path = f"{base_path}/{directory}"
        for file in os.listdir(path):

            if file.endswith(".jpg"):
                read_image_and_import_into_roboflow(f"{path}/{file}",file,dataset_name,os.environ['ROBOFLOW_API_KEY'])

            if file.endswith(".txt"):
                import_image_annotation(f"{path}/{file}",file,dataset_name,os.environ['ROBOFLOW_API_KEY'])