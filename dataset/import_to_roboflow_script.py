from dotenv import load_dotenv
from import_to_roboflow_functions import read_image_from

if __name__ == "__main__":

    load_dotenv('dataset.env')
    read_image_from("faces_and_plates_dataset","pco_dataset")

