import pandas as pd
import numpy as np

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import requests
import ipywidgets as widgets
from google.colab import output
from ultralytics import YOLO
from roboflow import Roboflow

from IPython.display import display, Javascript
from PIL import Image
import base64
import io
from io import BytesIO
import cv2

def filter_and_count(data, threshold=0.5, class_var="class"):
    filtered_data = [item for item in data if item["confidence"] >= threshold]
    result = {}
    for item in filtered_data:
        class_name = item[class_var]
        result[class_name] = result.get(class_name, 0) + 1
    return result

# Callback para procesar la imagen capturada
def process_photo(data_url):
    global photo
    image_data = base64.b64decode(data_url.split(',')[1])
    photo = Image.open(io.BytesIO(image_data))
    print("Photo captured and stored in the variable 'photo'.")

# Función para lanzar el JavaScript y capturar la foto
def take_photo(_):
    display(Javascript('''
        (async function() {
            const div = document.createElement('div');
            const video = document.createElement('video');
            const canvas = document.createElement('canvas');
            const button = document.createElement('button');
            const switchButton = document.createElement('button');
            let currentDeviceIndex = 0;
            let stream;

            // Agregar elementos al DOM
            document.body.appendChild(div);
            div.appendChild(video);
            div.appendChild(button);
            div.appendChild(switchButton);

            button.textContent = 'Take Photo';
            switchButton.textContent = 'Switch Camera';
            switchButton.style.marginTop = '10px';
            button.style.marginTop = '10px';

            // Función para obtener las cámaras disponibles
            const getVideoDevices = async () => {
                const devices = await navigator.mediaDevices.enumerateDevices();
                return devices.filter(device => device.kind === 'videoinput');
            };

            // Función para iniciar la cámara
            async function startCamera(deviceIndex) {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop()); // Detener flujo anterior si existe
                }

                const devices = await getVideoDevices();
                const constraints = {
                    video: {
                        deviceId: devices[deviceIndex].deviceId
                    }
                };

                stream = await navigator.mediaDevices.getUserMedia(constraints);
                video.srcObject = stream;
                video.play();
            }

            // Obtener las cámaras disponibles y iniciar la primera
            const devices = await getVideoDevices();
            await startCamera(currentDeviceIndex);

            // Evento para tomar la foto
            button.onclick = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const context = canvas.getContext('2d');
                context.drawImage(video, 0, 0);
                const dataURL = canvas.toDataURL('image/png');

                // Detener la cámara
                stream.getTracks().forEach(track => track.stop());
                div.remove();

                // Enviar la imagen capturada a Python
                google.colab.kernel.invokeFunction('notebook.process_photo', [dataURL], {});
            };

            // Evento para cambiar de cámara
            switchButton.onclick = async () => {
                currentDeviceIndex = (currentDeviceIndex + 1) % devices.length;
                await startCamera(currentDeviceIndex); // Cambiar a la siguiente cámara
            };
        })();
    '''))

# Firebase credentials
cred = credentials.Certificate('/content/drive/MyDrive/Colab Notebooks/Data/object-detection-credentials.json')

with open('/content/drive/MyDrive/Colab Notebooks/Data/firebase_credentials.json', 'r') as json_file:
    fb_cred = json.load(json_file)
    db_schema_name_str = fb_cred["db_schema_name"]
    project = fb_cred["project"]
    fb_api_key = fb_cred["api_key"]
    version_str = fb_cred["version"]

url = "https://raw.githubusercontent.com/martgnz/bcn-geodata/master/seccio-censal/seccio-censal.geojson" # URL del archivo GeoJSON en GitHub

# Roboflow Credentials
rf = Roboflow(api_key=fb_api_key)
project = rf.workspace().project(project)
model_roboflow = project.version(version_str).model

# Models Initialization
yolo_models_dict = {
    "custom_model": YOLO("/content/drive/MyDrive/Colab Notebooks/Data/best.pt"),
    "roboflow_model": project.version("2").model,
}
