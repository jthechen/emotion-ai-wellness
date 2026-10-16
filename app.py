import streamlit as st
import torch
import timm
from PIL import Image
import os
from google import genai


# =========================================================
# 1. CONFIGURACIÓN DE PÁGINA
# =========================================================

st.set_page_config(
    page_title="EmotionAI - Reconocimiento de Expresiones",
    page_icon="🧠",
    layout="centered"
)


# =========================================================
# 2. ESTILO VISUAL - TEMÁTICA EMOCIONES Y BIENESTAR
# =========================================================

st.markdown("""
<style>

/* Fondo general */
.stApp {
    background: linear-gradient(
        180deg,
        #FCFAFF 0%,
        #FFF9FB 55%,
        #FAFCFF 100%
    );
}

/* Contenedor principal */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 950px;
}

/* Títulos */
h1, h2, h3 {
    color: #35323F;
}

/* Cabecera principal */
.hero {
    padding: 30px 32px;
    border-radius: 22px;
    background: linear-gradient(
        135deg,
        #EFE8FF 0%,
        #FFE9F0 100%
    );
    border: 1px solid #E7DDF3;
    margin-bottom: 22px;
    box-shadow: 0px 6px 20px rgba(85, 63, 105, 0.07);
}

.hero-title {
    font-size: 44px;
    font-weight: 750;
    color: #35323F;
    margin-bottom: 5px;
}

.hero-subtitle {
    font-size: 18px;
    color: #625D6A;
    margin: 0;
    line-height: 1.5;
}

/* Tarjetas métricas */
div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E5DDEC;
    padding: 16px;
    border-radius: 16px;
    box-shadow: 0px 3px 12px rgba(70, 50, 90, 0.05);
}

/* Valor de las métricas */
div[data-testid="stMetricValue"] {
    font-size: 28px;
}

/* Expanders */
div[data-testid="stExpander"] {
    background-color: rgba(255, 255, 255, 0.85);
    border-radius: 15px;
    border: 1px solid #E8E1EF;
}

/* Carga de archivo */
div[data-testid="stFileUploader"] {
    background-color: #FFFFFF;
    border-radius: 16px;
}

/* Botón principal */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    font-weight: 650;
    padding: 0.7rem;
    border: 1px solid #CCB8E2;
}

/* Separadores */
hr {
    margin-top: 2rem;
    margin-bottom: 2rem;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. CABECERA TEMÁTICA
# =========================================================

# Se construye en una sola cadena para evitar que Streamlit
# muestre el HTML como bloque de código.
st.markdown(
    '<div class="hero">'
    '<div class="hero-title">🧠 EmotionAI</div>'
    '<div class="hero-subtitle">'
    'Explora expresiones faciales con inteligencia artificial '
    'y recibe una interpretación educativa y responsable.'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "EmotionAI analiza una imagen facial y estima si se aproxima a las "
    "categorías **depressive** o **natural**. Además, muestra el nivel de "
    "confianza del modelo y puede generar una orientación educativa "
    "relacionada con el bienestar."
)


# =========================================================
# 4. DESEMPEÑO GLOBAL DEL MODELO
# =========================================================

with st.expander(
    "📊 Ver desempeño global del modelo en conjunto de prueba"
):

    st.markdown(
        "**Modelo utilizado:** EfficientNet-B0 mediante Transfer Learning"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Accuracy en Test",
            "66.32%"
        )

    with col2:
        st.metric(
            "Accuracy balanceada en Test",
            "65.91%"
        )

    st.caption(
        "Estas métricas representan el rendimiento general del modelo "
        "sobre el conjunto de prueba. Son distintas de la confianza "
        "obtenida al analizar una imagen específica."
    )


# =========================================================
# 5. CARGA DEL MODELO
# =========================================================

DISP = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def cargar_modelo():

    # Misma arquitectura utilizada durante el entrenamiento
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

    if not os.path.exists(ruta):
        raise FileNotFoundError(
            "No se encontró el archivo efficientnet_emociones.pt"
        )

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

    # Mismo preprocesamiento utilizado por EfficientNet
    prep = timm.data.create_transform(
        **timm.data.resolve_model_data_config(modelo)
    )

    return modelo, clases, prep


modelo, CLASES, prep = cargar_modelo()


# =========================================================
# 6. FUNCIÓN DE RECOMENDACIÓN CON GEMINI
# =========================================================

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
        model="gemini-3.8-flash",
        contents=prompt
    )

    return respuesta.text


# =========================================================
# 7. CARGA DE IMAGEN
# =========================================================

st.markdown("## 😊 Analiza una expresión facial")

st.write(
    "Sube una fotografía facial en formato JPG, JPEG o PNG "
    "para realizar el análisis."
)

archivo = st.file_uploader(
    "Selecciona una imagen facial",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# =========================================================
# 8. INFERENCIA
# =========================================================

if archivo is not None:

    imagen = Image.open(
        archivo
    ).convert("RGB")

    # Mismo preprocesamiento utilizado durante el entrenamiento
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


    # =====================================================
    # 9. IMAGEN Y RESULTADO
    # =====================================================

    st.divider()

    col_img, col_resultado = st.columns([1, 1])

    with col_img:

        st.markdown("### 🖼️ Imagen analizada")

        st.image(
            imagen,
            use_container_width=True
        )

    with col_resultado:

        st.markdown("### 🧠 Lectura del modelo")

        st.metric(
            "Clase predicha",
            clase_predicha
        )

        st.metric(
            "Confianza de esta imagen",
            f"{confianza:.2%}"
        )

        # Interpretación visual de confianza
        if confianza >= 0.80:

            st.success(
                "✅ Nivel de confianza alto"
            )

        elif confianza >= 0.60:

            st.warning(
                "⚠️ Nivel de confianza moderado"
            )

        else:

            st.error(
                "⚠️ Nivel de confianza bajo"
            )

        st.caption(
            "La confianza corresponde únicamente a esta imagen. "
            "No representa el desempeño global del modelo."
        )


    # =====================================================
    # 10. PROBABILIDADES POR CLASE
    # =====================================================

    st.divider()

    st.markdown(
        "## 🎭 Probabilidades estimadas por clase"
    )

    st.write(
        "El gráfico muestra cómo se distribuye la probabilidad "
        "calculada por el modelo entre ambas categorías."
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


    # =====================================================
    # 11. ADVERTENCIA SEGÚN CONFIANZA
    # =====================================================

    if confianza < 0.60:

        st.warning(
            "⚠️ La confianza del modelo es inferior al 60%. "
            "El resultado es incierto y debe interpretarse con "
            "precaución. Se recomienda revisión humana."
        )


    # =====================================================
    # 12. RECOMENDACIÓN GENERADA POR IA
    # =====================================================

    st.divider()

    st.markdown(
        "## 💬 Orientación de bienestar"
    )

    st.write(
        "Puedes solicitar una orientación educativa generada por IA. "
        "Para ello se envían a Gemini únicamente la clase predicha "
        "y el nivel de confianza; la imagen no se envía al modelo de lenguaje."
    )

    if st.button(
        "✨ Generar orientación con IA"
    ):

        try:

            with st.spinner(
                "Generando una orientación responsable..."
            ):

                rec = generar_recomendacion(
                    clase_predicha,
                    confianza
                )

            st.success(
                "Orientación generada"
            )

            st.write(rec)

        except Exception as e:

            st.error(
                f"No fue posible generar la orientación: {e}"
            )


else:

    st.info(
        "👆 Sube una imagen facial para comenzar el análisis."
    )


# =========================================================
# 13. USO RESPONSABLE
# =========================================================

st.divider()

st.markdown(
    "## 🌿 Uso responsable"
)

st.info(
    "EmotionAI tiene exclusivamente fines educativos y experimentales. "
    "El análisis se basa únicamente en una expresión facial y no constituye "
    "un diagnóstico médico, psicológico ni de salud mental. "
    "Una expresión facial por sí sola no permite determinar el estado "
    "emocional o clínico de una persona. Si existe preocupación por el "
    "bienestar emocional, se recomienda consultar con un profesional."
)
