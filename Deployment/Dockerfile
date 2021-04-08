FROM oza6ut0ne/opencv
WORKDIR /soilClassifier/
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
ENTRYPOINT ["python", "server.py"]