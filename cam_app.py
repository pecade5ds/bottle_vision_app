import streamlit as st

# Main function for the application
def main():
    st.title("Capture a Photo from the Camera 📸")

    # Option to enable or disable the camera
    enable_camera = st.checkbox("Enable Camera")

    # Widget to capture the photo
    picture = st.camera_input("Take a photo", disabled=not enable_camera)

    # Display the captured photo if available
    if picture:
        st.image(picture, caption="Captured Photo")
        st.success("Photo captured successfully!")

if __name__ == "__main__":
    main()
