# Danone Waters Bottle Vision

## Overview

This project leverages **Computer Vision** to detect and categorize water bottles using images captured via a webcam. The application utilizes a **YOLO** model, integrated with **Roboflow** for pre-trained object detection, to classify different types of water bottles. The predictions are then stored in **Firebase Firestore** along with metadata such as store information and geographic location.

This tool can help Danone analyze product distribution in real-time and track inventory in stores efficiently.

---

## Features

- **Real-time Object Detection**: Detects and categorizes water bottles captured from a camera using a YOLO model.
- **Firebase Integration**: Saves predictions to Firebase Firestore, including metadata like postal code, store name, shelf ID, and geographical coordinates.
- **Geolocation**: Retrieves user location using IP or Geocoder to track the origin of predictions.
- **User-friendly Interface**: Built with **Streamlit** for easy interaction, enabling users to upload images, view predictions, and save results.

---

## Installation

To run this project, follow these steps:

### Prerequisites

- Python 3.8 or higher
- Firebase account and Firestore setup
- Roboflow account for the model API key
- config json for right query location.

---
### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/danone-bottle-vision.git
   cd danone-bottle-vision


2. Set up a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

4. Configure Firebase:

Create a Firebase project and set up Firestore.
Download the Firebase credentials JSON file and store it in the project directory.
In the Streamlit secrets (st.secrets["firebase"]), add the necessary Firebase credentials and API key.

5. Set up Roboflow API:

Create a project on Roboflow, upload the dataset, and generate an API key.
Store the API key and project details in the st.secrets["firebase"] configuration.

---

## Usage
Start the Streamlit application with streamlit run app.py.

The web app will open, allowing users to:

- Enable the camera to take a photo of a bottle.
- Automatically predict the category of the bottle using the YOLO model.
- Display the detected labels and their counts.
- Input metadata such as postal code, store name, and shelf ID.
- Save the predictions to Firebase for further analysis.
---
## Code Overview

1. Firebase Integration
The app connects to Firebase using the credentials stored in the st.secrets["firebase"] section, allowing it to save predictions and metadata to Firestore.

2. Roboflow Model
The project uses a pre-trained YOLO model available via Roboflow. It performs real-time object detection on the images captured from the user's camera.

3. Geolocation
The app tries to determine the user's geographical location using two methods:

* **Geocoder**: Fetches location data using IP.
* **ipapi.co**: A fallback method in case Geocoder fails.

4. Image Capture and Prediction

* Users can capture images of water bottles via a webcam.
* YOLO-based predictions are run on the captured image to detect the bottles and categorize them.
* The results are displayed along with metadata in the app interface.

Example Output:
Once an image is captured, the app will display the detected labels and their respective counts in a table. For example:

<div align="center">

| Label  | Count |
|--------|-------|
| Water  | 3     |
| Bottle | 2     |

</div>

It will also display the user's geographical location and provide input fields for metadata like store name, postal code, and shelf ID. The predictions can be saved to Firestore.

---
## License
This project is licensed under the MIT License - see the LICENSE file for details.

---
## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

---
## Acknowledgments
Roboflow for providing the pre-trained YOLO model.
Streamlit for the easy-to-use framework for building interactive applications.
Firebase for real-time database services.
