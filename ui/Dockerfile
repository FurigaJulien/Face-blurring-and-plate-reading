FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8501
RUN wget https://github.com/ultralytics/yolov5/archive/refs/heads/master.zip -P ./app \
  && unzip ./app/master.zip
ENTRYPOINT ["streamlit", "run"]


CMD ["app/app.py"]