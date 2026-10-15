import streamlit as st
import torch
import timm
from PIL import Image
import os
from google import genai

# 1. Configuración de página
st.set_page_config(
    page_title="EmotionAI - Reconocimiento de Expresiones",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 EmotionAI")
st.write(
    "Aplicación educativa para clasificar expresiones faciales "
    "entre las categorías **depressive** y **natural**."
)

# 2. Desempeño global del modelo en test (Punto requerido por el certamen)
with st.expander("ℹ️ Ver métricas globales del modelo en conjunto de prueba"):
    st.markdown("""
    - **Modelo:** EfficientNet-B0 (Transfer Learning)
    - **Accuracy en Test:** 74.08%
    - **Accuracy Balanceada en Test:** 73.84%
    *(Estas métricas representan el rendimiento general del modelo en datos nunca antes vistos, a diferencia de la confianza específica de cada imagen).*
    """)

# 3. Carga del modelo en caché
DISP = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def cargar_modelo():
    modelo = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    
    # Buscar el checkpoint en la ruta local o en /content/
    ruta = "efficientnet_emociones.pt"
    if not os.path.exists(ruta) and os.path.exists("/content/efficientnet_emociones.pt"):
        ruta = "/content/efficientnet_emociones.pt"
        
    checkpoint = torch.load(ruta, map_location=DISP)
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.to(DISP)
    modelo.eval()
    
    clases = checkpoint["clases"]
    prep = timm.data.create_transform(**timm.data.resolve_model_data_config(modelo))
    return modelo, clases, prep

modelo, CLASES, prep = cargar_modelo()

# 4. Función de recomendación con Gemini
def generar_recomendacion(clase, confianza):
    # Obtener API key desde st.secrets o variable de entorno
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", None))
    if not api_key:
        raise ValueError("No se encontró GEMINI_API_KEY en st.secrets ni en las variables de entorno.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
    Actúa como un asistente educativo de bienestar.
    El resultado de un modelo de clasificación de imágenes fue:
    - Clase predicha: {clase}
    - Confianza: {confianza:.2%}

    Genera una recomendación breve, clara y responsable.
    Reglas:
    - No realices diagnósticos de depresión ni de salud mental.
    - No afirmes que la persona tiene una condición clínica.
    - Explica que el resultado proviene solo de una imagen facial.
    - Si la confianza es menor al 60%, indica que el resultado es incierto.
    - Recomienda consultar a un profesional si existe preocupación real por el bienestar emocional.
    - Mantén la respuesta en un máximo de 4 frases.
    """

    respuesta = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return respuesta.text

# 5. Carga de imagen e inferencia
archivo = st.file_uploader("Sube una imagen facial", type=["jpg", "jpeg", "png"])

if archivo is not None:
    imagen = Image.open(archivo).convert("RGB")
    st.image(imagen, caption="Imagen cargada", width=300)

    # Inferencia
    x = prep(imagen).unsqueeze(0).to(DISP)
    with torch.no_grad():
        salida = modelo(x)
        probabilidades = torch.softmax(salida, dim=1)[0]

    indice = probabilidades.argmax().item()
    clase_predicha = CLASES[indice]
    confianza = probabilidades[indice].item()

    st.subheader("Resultado de la clasificación")
    st.write(f"**Clase predicha:** `{clase_predicha}`")
    st.write(f"**Confianza:** `{confianza:.2%}`")

    # Gráfico de probabilidades
    st.subheader("Probabilidades por clase")
    prob_dict = {CLASES[i]: float(probabilidades[i]) for i in range(len(CLASES))}
    st.bar_chart(prob_dict)

    # Advertencia si la confianza es baja
    if confianza < 0.60:
        st.warning("⚠️ La confianza del modelo es baja (< 60%). Este resultado es incierto y requiere revisión humana.")

    # 6. Botón de recomendación con IA
    st.divider()
    st.subheader("Recomendación generada por IA")
    if st.button("Generar recomendación"):
        try:
            with st.spinner("Generando recomendación responsable con Gemini..."):
                rec = generar_recomendacion(clase_predicha, confianza)
            st.success("Recomendación generada:")
            st.write(rec)
        except Exception as e:
            st.error(f"No fue posible generar la recomendación: {e}")
else:
    st.info("👆 Por favor sube una imagen facial para comenzar el análisis.")

# 7. Descargo de responsabilidad final
st.divider()
st.caption(
    "⚠️ **Aviso responsable:** Esta aplicación tiene exclusivamente fines educativos y experimentales. "
    "El análisis se basa únicamente en una expresión facial y no constituye bajo ninguna circunstancia un diagnóstico clínico o psicológico."
)
