import streamlit as st
import geocoder
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

# # Init credentials
# initialize_app(cred)

# if not firebase_admin._apps:
#     initialize_app(cred)
#     db = firestore.client()

# # Connect to Firestore
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

def filter_and_count(data, threshold=0.5, class_var="class"):
    filtered_data = [item for item in data if item["confidence"] >= threshold]
    result = {}
    for item in filtered_data:
        class_name = item[class_var]
        result[class_name] = result.get(class_name, 0) + 1
    return result

# def get_lat_lon():
#     # Use geocoder to get the location based on IP
#     g = geocoder.ip('me')
    
#     if g.ok:
#         lat = g.latlng[0]  # Latitude
#         lon = g.latlng[1]  # Longitude
#         return lat, lon
#     else:
#         st.error("Could not retrieve location from IP address.")
#         return None, None

# Main function for the application
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

        st.markdown("---")

        # get coordinates
        # lat_long_vec = get_lat_lon()
        
        # Show detected labels and counts
        st.write("Predicted labels and counts:", robo_detected_label_counts_dict)
        # st.write("coord:", lat_long_vec)
        
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
            if not postal_code or not store_name:
                st.warning("Please fill in all fields before saving the predictions.")
            else:
                try:
                    # Save predictions to Firebase
                    doc_ref = db.collection(db_schema_name_str).add({
                        "predictions": robo_detected_label_counts_dict,
                        "post_code": postal_code,
                        "shelf id": shelf_id,
                        "store_name": store_name,
                        # "coordinates": lat_long_vec,
                        "photo_base64": base64.b64encode(img_bytes).decode("utf-8"),
                    })

                    st.success(f"Predictions successfully saved with ID: {doc_ref[1].id}!")

                except Exception as e:
                    st.error(f"An error occurred while saving predictions: {e}")

if __name__ == "__main__":
    main()
