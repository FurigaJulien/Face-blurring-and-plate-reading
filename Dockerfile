FROM python:3.9
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
COPY . /app
RUN wget https://github.com/ultralytics/yolov5/archive/refs/heads/master.zip -P ./app \
  && unzip ./app/master.zip
ENTRYPOINT ["streamlit", "run"]


CMD ["app/app.py"]