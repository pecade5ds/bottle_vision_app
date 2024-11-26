import streamlit as st
from streamlit_js_eval import streamlit_js_eval
from firebase_admin import credentials, initialize_app, firestore
from io import BytesIO
import base64
import json
import cv2
import numpy as np
from PIL import Image
from roboflow import Roboflow

# Firebase credentials
cred = credentials.Certificate('./credentials/object-detection-credentials.json')

with open('./credentials/firebase_credentials.json', 'r') as json_file:
    fb_cred = json.load(json_file)
    db_schema_name_str = fb_cred["db_schema_name"]
    project = fb_cred["project"]
    fb_api_key = fb_cred["api_key"]
    version_str = fb_cred["version"]

# Roboflow Credentials
rf = Roboflow(api_key=fb_api_key)
project = rf.workspace().project(project)
model_roboflow = project.version(version_str).model

# Models Initialization
yolo_models_dict = {
    # "custom_model": YOLO("/content/drive/MyDrive/Colab Notebooks/Data/best.pt"),
    "roboflow_model": project.version("2").model,
}

def filter_and_count(data, threshold=0.5, class_var="class"):
    filtered_data = [item for item in data if item["confidence"] >= threshold]
    result = {}
    for item in filtered_data:
        class_name = item[class_var]
        result[class_name] = result.get(class_name, 0) + 1
    return result

def get_lat_lon():
    # Execute geolocation from the browser
    location = streamlit_js_eval(
        js_expressions="await new Promise(resolve => navigator.geolocation.getCurrentPosition(pos => resolve(pos.coords)))",
        key="geo_eval",
    )
    
    if location:
        lat = location.get("latitude", "Unavailable")
        lon = location.get("longitude", "Unavailable")
        return lat, lon
    else:
        st.error("Could not retrieve location. Make sure to allow location access in your browser.")
        return None, None

# Main function for the application
def main():
    st.title("Danone - Waters Bottle Vision 📸")

    # Option to enable or disable the camera
    enable_camera = st.checkbox("Enable Camera")

    # Widget to capture the photo
    picture = st.camera_input("Take a photo", disabled=not enable_camera)

    # Display the captured photo if available
    if picture:
        st.success("Photo captured successfully!")
        
        # get coordinates
        lat_long_vec = get_lat_lon()
        
        # Load the picture into a PIL Image
        img = Image.open(BytesIO(picture.getvalue()))
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        # Convert image to numpy array
        image_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Predict using YOLO model
        roboflow_result = yolo_models_dict["roboflow_model"].predict(image_np, confidence=50, overlap=30)
        robo_detected_label_counts_dict = filter_and_count(roboflow_result.json()["predictions"], threshold=0.5, class_var="class")

        # Show detected labels and counts
        st.write("Predicted labels and counts:", robo_detected_label_counts_dict)
        st.write("coord:", lat_long_vec)
        
        # Section to save predictions in Firebase
        st.subheader("Save Predictions to Firebase")
        
        # Input fields for postal code and store name
        postal_code = st.text_input("Enter Postal Code:", placeholder="Example: 12345")
        store_name = st.text_input("Enter Store Name:", placeholder="Example: XYZ Store")
        
        # Button to save predictions
        if st.button("Save Predictions"):
            if not postal_code or not store_name:
                st.warning("Please fill in all fields before saving the predictions.")
            else:
                try:
                    # Save predictions to Firebase
                    # doc_ref = db.collection("predictions").add({
                    #     "predictions": robo_detected_label_counts_dict,
                    #     "post_code": postal_code,
                    #     "store_name": store_name,
                    #     "coordinates": f"{lat}, {lon}" if lat and lon else "Location unavailable",
                    #     "photo_base64": base64.b64encode(img_bytes).decode("utf-8"),
                    # })
                    doc_ref = [0,1]
                    st.success(f"Predictions successfully saved with ID: {doc_ref[1].id}!")
                except Exception as e:
                    st.error(f"An error occurred while saving predictions: {e}")

if __name__ == "__main__":
    main()
