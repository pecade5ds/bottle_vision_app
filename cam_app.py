import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
import cv2
import numpy as np

# Configuración para detectar las cámaras disponibles
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

photo = None  # Variable global para almacenar la foto

# Procesador de video para capturar el frame actual
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.capture_frame = False
        self.captured_image = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convertir el frame a una matriz numpy
        if self.capture_frame:
            self.captured_image = img  # Guardar la imagen capturada
            self.capture_frame = False  # Resetear el flag
        return img

# Función principal de la aplicación
def main():
    global photo

    st.title("Captura de Foto desde la Cámara")

    # Desplegar la cámara en la app
    webrtc_ctx = webrtc_streamer(
        key="camera",
        mode="sendrecv",
        client_settings=ClientSettings(
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        ),
        video_transformer_factory=VideoProcessor,
    )

    if webrtc_ctx.video_transformer:
        st.info("Haz clic en 'Capturar Foto' para tomar una foto.")
        capture_button = st.button("Capturar Foto")

        # Botón para capturar la foto
        if capture_button:
            webrtc_ctx.video_transformer.capture_frame = True
            st.success("Foto capturada correctamente.")
            if webrtc_ctx.video_transformer.captured_image is not None:
                # Guardar la foto capturada en la variable global
                photo = webrtc_ctx.video_transformer.captured_image
                st.image(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB), caption="Foto Capturada")

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
