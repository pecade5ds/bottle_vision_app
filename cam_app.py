import streamlit as st
import requests
from typing import Optional, Tuple
import firebase_admin
from firebase_admin import credentials, initialize_app, firestore
from io import BytesIO
import base64
import json
import geocoder
import numpy as np
from PIL import Image
from roboflow import Roboflow
from utils import *

# Firebase credentials

db = firestore.Client.from_service_account_json('./credentials/bottlevision-credentials.json')

# cred = credentials.Certificate('./credentials/bottlevision-credentials.json')

# if not firebase_admin._apps:
#     initialize_app(cred)

# # # Connect to Firestore
# db = firestore.client()

with open('./credentials/robo_credentials.json', 'r') as json_file:
    robo_cred = json.load(json_file)
    db_schema_name_str = robo_cred["db_schema_name"]
    project = robo_cred["project"]
    fb_api_key = robo_cred["api_key"]
    version_str = robo_cred["version"]

# Roboflow Credentials
rf = Roboflow(api_key=fb_api_key)
project = rf.workspace().project(project)
model_roboflow = project.version(version_str).model

# Models Initialization
yolo_models_dict = {
    # "custom_model": YOLO("/content/drive/MyDrive/Colab Notebooks/Data/best.pt"),
    "roboflow_model": project.version("2").model,
}

def get_location_geocoder() -> Tuple[Optional[float], Optional[float]]:
    """
    Get location using geocoder library
    """
    g = geocoder.ip('me')
    if g.ok:
        return g.latlng[0], g.latlng[1]
    return None, None

def get_location_ipapi() -> Tuple[Optional[float], Optional[float]]:
    """
    Fallback method using ipapi.co service
    """
    try:
        response = requests.get('https://ipapi.co/json/')
        if response.status_code == 200:
            data = response.json()
            lat = data.get('latitude')
            lon = data.get('longitude')
            
            if lat is not None and lon is not None:
                # Store additional location data in session state
                st.session_state.location_data = {
                    'city': data.get('city'),
                    'region': data.get('region'),
                    'country': data.get('country_name'),
                    'ip': data.get('ip')
                }
                return lat, lon
    except requests.RequestException as e:
        st.error(f"Error retrieving location from ipapi.co: {str(e)}")
    return None, None

def get_location() -> Tuple[Optional[float], Optional[float]]:
    """
    Tries to get location first using geocoder, then falls back to ipapi.co
    """
    # Try geocoder first
    lat, lon = get_location_geocoder()
    
    # If geocoder fails, try ipapi
    if lat is None:
        st.info("Primary geolocation method unsuccessful, trying alternative...")
        lat, lon = get_location_ipapi()
    
    return lat, lon

def main():
    st.title("Danone - Waters Bottle Vision 📸")
    # Custom horizontal divider
    st.markdown(
        """
        <style>
        hr {
            border: 2px solid #FF6347;
            border-radius: 5px;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    # Example of using a custom divider
    st.markdown("---")

    # Option to enable or disable the camera
    enable_camera = st.checkbox("Enable Camera")

    # Widget to capture the photo
    picture = st.camera_input("Take a photo", disabled=not enable_camera)

    # Display the captured photo if available
    if picture:
        st.success("Photo captured successfully!")

        # Predict using YOLO model
        roboflow_result = yolo_models_dict["roboflow_model"].predict(np.array(Image.open(BytesIO(picture.getvalue()))) , confidence=50, overlap=30)
        robo_detected_label_counts_dict = filter_and_count(roboflow_result.json()["predictions"], threshold=0.5, class_var="class")

        with st.spinner("Retrieving your location..."):
            lat, lon = get_location()
        
        st.write(f"{(lat, lon)}")
        st.markdown("---")
        
        # Show detected labels and counts
        if robo_detected_label_counts_dict:
            st.write("Predicted labels and counts:")
            st.table(pd.DataFrame(list(robo_detected_label_counts_dict.items()), columns=["Label", "Count"])) 
        else:
            st.write("No predicted labels to display.")
            
        # Section to save predictions in Firebase
        st.subheader("Save Predictions to Firebase")
        
        # Input fields for postal code and store name
        col1, col2, col3 = st.columns([1, 1, 1])  # Adjust the column width ratio
        
        # Input fields for postal code and store name in parallel
        with col1:
            postal_code = st.text_input("Enter Postal Code:", placeholder="Example: 12345", key="postal_code_input")
        
        with col2:
            store_name = st.text_input("Enter Store Name:", placeholder="Example: XYZ Store", key="store_name_input")

        with col3:
            shelf_id = st.text_input("Enter Shelf id:", placeholder="Example: 1", key="shelf_id_input")

        if st.button("Save Predictions"):

            try:
                # Save predictions to Firebase
                doc_ref = db.collection(db_schema_name_str).add(
                    {
                    "predictions": robo_detected_label_counts_dict,
                    "post_code": postal_code,
                    "shelf id": shelf_id,
                    "store_name": store_name,
                    "coordinates": (lat, lon),
                    # "photo_base64": base64.b64encode(picture).decode("utf-8"),
                }
                )

                st.success(f"Predictions successfully saved with ID: {doc_ref[1].id}!")

            except Exception as e:
                st.error(f"An error occurred while saving predictions: {e}")

if __name__ == "__main__":
    main()
