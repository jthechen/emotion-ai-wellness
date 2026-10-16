import streamlit as st
import torch
import timm
from PIL import Image
import os
from google import genai


#1. configuracion pagina

st.set_page_config(
    page_title="EmotionAI - Reconocimiento de Expresiones",
    page_icon="🧠",
    layout="centered"
)

#2. estilo visual

st.markdown("""
<style>

/* Espaciado general */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Títulos */
h1, h2, h3 {
    color: #30313D;
}

/* Tarjetas */
div[data-testid="stMetric"] {
    background-color: #F8F7FB;
    border: 1px solid #E6E2EC;
    padding: 15px;
    border-radius: 14px;
}

/* Botón principal */
.stButton > button {
    width: 100%;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem;
}

</style>
""", unsafe_allow_html=True)


#3. cabecera app

st.markdown("""
<div style="
    padding: 22px;
    border-radius: 18px;
    background: linear-gradient(135deg, #F4F0FF 0%, #FFF2F5 100%);
    margin-bottom: 22px;
">
    <h1 style="margin-bottom:5px;">🧠 EmotionAI</h1>

    <p style="
        font-size:17px;
        margin:0;
        color:#555;
    ">
        Clasificación educativa de expresiones faciales mediante
        inteligencia artificial.
    </p>
</div>
""", unsafe_allow_html=True)

st.write(
    "La aplicación clasifica imágenes faciales entre las categorías "
    "**depressive** y **natural**, mostrando la predicción, "
    "su confianza y una recomendación educativa generada por IA."
)


#4. desempeño global del modelo

with st.expander(
    "📊 Ver métricas globales del modelo en conjunto de prueba"
):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Modelo",
            "EfficientNet-B0"
        )

    with col2:
        st.metric(
            "Accuracy Test",
            "74.08%"
        )

    with col3:
        st.metric(
            "Accuracy balanceada",
            "73.84%"
        )

    st.caption(
        "Estas métricas representan el rendimiento general del modelo "
        "sobre datos del conjunto de prueba. No deben confundirse con "
        "la confianza obtenida para una imagen específica."
    )


#5. carga del modelo

DISP = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def cargar_modelo():

    modelo = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=2
    )

    # Buscar checkpoint local o en /content/
    ruta = "efficientnet_emociones.pt"

    if (
        not os.path.exists(ruta)
        and os.path.exists("/content/efficientnet_emociones.pt")
    ):
        ruta = "/content/efficientnet_emociones.pt"

    checkpoint = torch.load(
        ruta,
        map_location=DISP
    )

    modelo.load_state_dict(
        checkpoint["model_state_dict"]
    )

    modelo.to(DISP)
    modelo.eval()

    clases = checkpoint["clases"]

    prep = timm.data.create_transform(
        **timm.data.resolve_model_data_config(modelo)
    )

    return modelo, clases, prep


modelo, CLASES, prep = cargar_modelo()


#6.funcion por gemini

def generar_recomendacion(clase, confianza):

    api_key = st.secrets.get(
        "GEMINI_API_KEY",
        os.environ.get("GEMINI_API_KEY", None)
    )

    if not api_key:
        raise ValueError(
            "No se encontró GEMINI_API_KEY."
        )

    client = genai.Client(
        api_key=api_key
    )

    prompt = f"""
    Actúa como un asistente educativo de bienestar.

    El resultado de un modelo de clasificación de imágenes fue:

    - Clase predicha: {clase}
    - Confianza: {confianza:.2%}

    Genera una recomendación breve, clara y responsable.

    Reglas:
    - No realices diagnósticos de depresión ni de salud mental.
    - No afirmes que la persona tiene una condición clínica.
    - Explica que el resultado proviene únicamente de una imagen facial.
    - Si la confianza es menor al 60%, indica que el resultado es incierto.
    - Recomienda consultar a un profesional si existe preocupación real
      por el bienestar emocional.
    - Mantén la respuesta en un máximo de 4 frases.
    """

    respuesta = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return respuesta.text


#7 carga de imagen 

st.subheader("📷 Analizar una imagen")

archivo = st.file_uploader(
    "Sube una imagen facial",
    type=["jpg", "jpeg", "png"]
)

#8. inferencia

if archivo is not None:

    imagen = Image.open(
        archivo
    ).convert("RGB")

    # Preprocesamiento del modelo
    x = prep(
        imagen
    ).unsqueeze(0).to(DISP)

    # Inferencia
    with torch.no_grad():

        salida = modelo(x)

        probabilidades = torch.softmax(
            salida,
            dim=1
        )[0]

    indice = probabilidades.argmax().item()

    clase_predicha = CLASES[indice]

    confianza = probabilidades[indice].item()


   #9. imagen y resultado

    col_img, col_resultado = st.columns(2)

    with col_img:

        st.markdown("### Imagen analizada")

        st.image(
            imagen,
            use_container_width=True
        )

    with col_resultado:

        st.markdown("### Resultado")

        st.metric(
            "Clase predicha",
            clase_predicha
        )

        st.metric(
            "Confianza",
            f"{confianza:.2%}"
        )

        if confianza >= 0.80:

            st.success(
                "✅ Confianza alta"
            )

        elif confianza >= 0.60:

            st.warning(
                "⚠️ Confianza moderada"
            )

        else:

            st.error(
                "⚠️ Confianza baja"
            )

        st.caption(
            "La confianza corresponde únicamente a esta imagen "
            "y no representa el desempeño global del modelo."
        )


 #10. probabilidades

    st.divider()

    st.subheader(
        "📊 Probabilidades por clase"
    )

    prob_dict = {
        CLASES[i]: float(probabilidades[i])
        for i in range(len(CLASES))
    }

    st.bar_chart(
        prob_dict
    )

    p1, p2 = st.columns(2)

    with p1:

        st.metric(
            CLASES[0],
            f"{float(probabilidades[0]):.2%}"
        )

    with p2:

        st.metric(
            CLASES[1],
            f"{float(probabilidades[1]):.2%}"
        )

#11. advertencia 

    if confianza < 0.60:

        st.warning(
            "La confianza del modelo es inferior al 60%. "
            "Este resultado debe interpretarse con precaución "
            "y se recomienda revisión humana."
        )


 #12. recomendacion generada por ia 
    st.divider()

    st.subheader(
        "💬 Recomendación generada por IA"
    )

    st.write(
        "Puedes solicitar una recomendación educativa "
        "basada en la clase predicha y su nivel de confianza."
    )

    if st.button(
        "✨ Generar recomendación"
    ):

        try:

            with st.spinner(
                "Generando recomendación responsable con Gemini..."
            ):

                rec = generar_recomendacion(
                    clase_predicha,
                    confianza
                )

            st.success(
                "Recomendación generada"
            )

            st.write(rec)

        except Exception as e:

            st.error(
                f"No fue posible generar la recomendación: {e}"
            )


else:

    st.info(
        "👆 Sube una imagen facial para comenzar el análisis."
    )


#13.aviso responsable

st.divider()

st.markdown("### ⚠️ Uso responsable")

st.caption(
    "Esta aplicación tiene exclusivamente fines educativos y experimentales. "
    "El análisis se basa únicamente en una expresión facial y no constituye "
    "bajo ninguna circunstancia un diagnóstico médico, psicológico o de "
    "salud mental. Si existe preocupación por el bienestar emocional de "
    "una persona, se recomienda consultar con un profesional."
)
