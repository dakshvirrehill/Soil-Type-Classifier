FROM oza6ut0ne/opencv
WORKDIR /soilClassifier/
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 4567
ENTRYPOINT ["python", "server.py"]