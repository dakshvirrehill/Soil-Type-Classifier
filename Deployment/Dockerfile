FROM python:3.8.5
WORKDIR /soilClassifier/
COPY . .
RUN pip install -r requirements.txt
EXPOSE 4567
ENTRYPOINT ["python", "server.py"]