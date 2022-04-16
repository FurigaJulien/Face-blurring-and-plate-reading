FROM python:3.9
WORKDIR /app
COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt
EXPOSE 8501
RUN wget https://github.com/ultralytics/yolov5/archive/refs/heads/master.zip -P ./app \
  && unzip ./app/master.zip
  
COPY . /app
ENTRYPOINT ["streamlit", "run"]


CMD ["app/app.py"]