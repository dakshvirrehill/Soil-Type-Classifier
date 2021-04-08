FROM jjanzic/docker-python3-opencv
WORKDIR /soilClassifier/
COPY . .
RUN pip install -r requirements.txt
EXPOSE 4567
ENTRYPOINT ["python", "server.py"]