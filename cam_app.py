import streamlit as st
from typing import Optional, Tuple
import firebase_admin
from firebase_admin import credentials, initialize_app, firestore
from io import BytesIO
from ultralytics import YOLO
import json
import numpy as np
from PIL import Image
from roboflow import Roboflow
from google.cloud import firestore
from utils import *

# Firebase credentials
db = firestore.Client.from_service_account_info(st.secrets["firebase"])

with open('./config/query_config.json', 'r') as json_file:
    db_schema_name_str = json.load(json_file)["db_schema_name"]

# Roboflow Credentials
rf = Roboflow(api_key=st.secrets["roboflow"]["api_key"])
project = rf.workspace().project(st.secrets["roboflow"]["project"])
model_roboflow = project.version(st.secrets["roboflow"]["version"]).model

# Models Initialization
yolo_models_dict = {
    # "custom_model": YOLO("/content/drive/MyDrive/Colab Notebooks/Data/best.pt"),
    "roboflow_model": project.version(st.secrets["roboflow"]["version"]).model,
}

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
        .toggle-button {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        .toggle-button div {
            cursor: pointer;
            padding: 10px 20px;
            background-color: #FF6347;
            color: white;
            border-radius: 25px;
            text-align: center;
            transition: background-color 0.3s;
        }
        .toggle-button div:hover {
            background-color: #FF4500;
        }
        .selected {
            background-color: #32CD32 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Divider
    st.markdown("---")

    # Toggle Button
    mode = st.session_state.get("mode", "Take a Photo")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📸 Take a Photo", key="take_photo", on_click=lambda: st.session_state.update({"mode": "Take a Photo"})):
            mode = "Take a Photo"
    
    with col2:
        if st.button("📂 Upload a Photo", key="upload_photo", on_click=lambda: st.session_state.update({"mode": "Upload a Photo"})):
            mode = "Upload a Photo"
    
    # Initialize variables
    picture = None

    if mode == "Take a Photo":
        # Option to enable or disable the camera
        enable_camera = st.checkbox("Enable Camera")

        # Widget to capture the photo
        picture = st.camera_input("Take a photo", disabled=not enable_camera)
        
    elif mode == "Upload a Photo":
        # Widget to upload a photo
        picture = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

    st.markdown(
        """<hr style="border: 1px solid #d3d3d3;"/>""", 
        unsafe_allow_html=True
    )
    
    # Display the captured photo if available
    if picture:
        st.success("Photo ready for processing!")
        picture = picture.read() # Byte data returned from the read method
        st.write(f"{type(picture)}")
        
        # Predict using YOLO model
        #################################
        roboflow_result = yolo_models_dict["roboflow_model"].predict(np.array(Image.open(picture)) , confidence=50, overlap=30)
        robo_detected_label_counts_dict = filter_and_count(roboflow_result.json()["predictions"], threshold=0.5, class_var="class")

        # Base model for Bottle detection (denominator for computing "Water store share")
        #################################
        model = YOLO('yolov8n.pt')  
        
        # Predict just on bottles        
        bottles_pred = model.predict(Image.open(io.BytesIO(byte_data)),
                                classes=[39],  # ID "bottle" class
                                conf=0.5)
        
        denominator_results = filter_and_count(bottles_pred[0].summary(), threshold=0.5, class_var="name")["bottle"]

        # with st.spinner("Retrieving your location..."):
        #     lat, lon = get_location(st)
        
        # st.write(f"{(lat, lon)}")
        
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
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])  # Adjust the column width ratio
        
        # Input fields for postal code and store name in parallel
        with col1:
            postal_code = st.text_input("Enter Postal Code:", placeholder="Example: 12345", key="postal_code_input")
        
        with col2:
            store_name = st.text_input("Enter Store Name:", placeholder="Example: XYZ Store", key="store_name_input")

        with col3:
            shelf_id = st.text_input("Enter Shelf id:", placeholder="Example: 1", key="shelf_id_input")
        
        with col4:
            store_type = st.selectbox("Select Store Type", ["TT", "OT"], key="store_type", index=1)

        with col5:
            photo_type = st.selectbox("Select Store Type", ["Test", "Prod"], key="foto_type", index=0) # This is when testing the app to filter garbage

        if st.button("Save Predictions"):
            try:
                # Save predictions to Firebase
                doc_ref = db.collection(db_schema_name_str).add(
                    {
                        "photo_type": photo_type,
                        "predictions": robo_detected_label_counts_dict,
                        "Num_bottles":denominator_results,
                        "post_code": postal_code,
                        "shelf id": shelf_id,
                        "store_type": store_type,
                        "store_name": store_name,
                        "coordinates": (lat, lon),
                        "photo": convert_image_to_base64(picture),
                }
                )

                st.success(f"Predictions successfully saved with ID: {doc_ref[1].id}!")

            except Exception as e:
                st.error(f"An error occurred while saving predictions: {e}")

if __name__ == "__main__":
    main()
