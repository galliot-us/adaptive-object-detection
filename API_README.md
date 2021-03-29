### Single Image Inference API

Single image inference API enables you to inference an image on CPUs or GPUs by listening on port 5000 and only accessible from localhost by default. For using the API follow the below instructions.

**1) Build Docker Image**
Make sure your system fulfills the prerequisites and then clone this repository to your local system by running this command:
```
git clone https://github.com/neuralet/adaptive-object-detection.git
cd adaptive-object-detection

# Build Docker image
docker build -f api.Dockerfile -t "single_img_api" .

# Run on CPUs
docker run -it -p PORT:5000 -v "$PWD":/repo single_img_api

# Run on GPUs
docker run --gpus all -it -p PORT:5000 -v "$PWD":/repo single_img_api
```
**1) API Description**
To make sure the API in running successfully you can call to check the health status of running service:
```
curl -X 'GET' \
  'http://127.0.0.1:PORT/api/ping' \
  -H 'accept: application/json'
```
If the server is running, the response is:
```
{
  "health": "healthy!"
}
```

For running inferencing API make a REST call to `/api/inference`:

```
curl -X 'POST' \
  'http://127.0.0.1:2028/api/inference' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@FILENAME.jpg;type=image/jpeg'
```
The result is JSON that tells you the object classes and boxes:
```
"[{\"category\": \"person\", \"bbox\": [0.7400702834129333, 0.7336881756782532, 0.9079803824424744, 0.9276494383811951], \"score\": 0.6597123742103577}]"
```
