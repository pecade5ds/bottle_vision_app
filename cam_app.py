import streamlit as st

# Función principal de la aplicación
def main():
    st.title("Captura de Foto desde la Cámara 📸")

    # Opción para habilitar o deshabilitar la cámara
    enable_camera = st.checkbox("Habilitar cámara")

    # Widget para capturar la foto
    picture = st.camera_input("Toma una foto", disabled=not enable_camera)

    # Mostrar la foto capturada si está disponible
    if picture:
        st.image(picture, caption="Foto Capturada")
        st.success("¡Foto capturada correctamente!")

if __name__ == "__main__":
    main()
