# ============================================================
#  CHATBOT EMOCIONAL TERAPÉUTICO + REPORTE PROFESIONAL
#  Versión mejorada con respuestas dinámicas de Gemini (TCC/DBT/ACT)
# ============================================================

import os
import json
import datetime
import requests # Necesario para la API
import time # Necesario para reintentos de API
from functools import lru_cache

import streamlit as st
import pandas as pd

# NLP
import nltk
# import spacy -> No se estaba usando en las reglas, se elimina para acelerar la carga
from transformers import pipeline

st.set_page_config(page_title="Chatbot Emocional", layout="centered")

# ---------------------------
# Config / paths
# ---------------------------
HISTORIAL_FILE = "historial_emocional.json"

# Configuración de la API de Gemini
# ¡SECURE! Leemos la clave de una variable de entorno, no la pegamos aquí.
API_KEY = os.environ.get("GEMINI_API_KEY") 
API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
MAX_RETRIES = 3 # Máximo de reintentos para el retroceso exponencial

# ---------------------------
# Recursos (descargas)
# ---------------------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# spaCy Spanish model
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("es_core_news_sm")
    except Exception:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "es_core_news_sm"], check=True)
        return spacy.load("es_core_news_sm")

# nlp = load_spacy() -> Eliminado

# Transformers sentiment pipeline
@st.cache_resource
def load_sentiment_pipeline():
    # Modelo multilingüe explícito que usa "estrellas" y entiende español
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    return pipeline("sentiment-analysis", model=model_name)
sentiment_pipe = load_sentiment_pipeline()

# ---------------------------
# Utilidades: Llamada a la API de Gemini con Retroceso Exponencial
# ---------------------------
def gemini_generate_response(system_prompt: str, chat_history: list, user_query: str):
    """
    Genera una respuesta terapéutica avanzada usando la API de Gemini,
    considerando el historial de chat.
    """
    
    # Formatear historial para la API
    history_payload = []
    for msg in chat_history:
        role = "user" if msg["sender"] == "user" else "model"
        history_payload.append({"role": role, "parts": [{"text": msg["text"]}]})

    # Añadir el query actual del usuario
    history_payload.append({"role": "user", "parts": [{"text": user_query}]})

    payload = {
        "contents": history_payload,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "temperature": 0.8,
            "topP": 0.9,
            "maxOutputTokens": 150,
        }
    }

    # Llamada a la API con Retroceso Exponencial
    for attempt in range(MAX_RETRIES):
        try:
            # Primero, verificar si la API_KEY existe
            if not API_KEY:
                print("Error: La variable de entorno GEMINI_API_KEY no está configurada.")
                return "Error de configuración: La clave de API no fue encontrada. Por favor, contacta al administrador."

            response = requests.post(f"{API_URL_BASE}?key={API_KEY}", 
                                     headers={'Content-Type': 'application/json'}, 
                                     json=payload,
                                     timeout=20) # Timeout para evitar esperas infinitas
            
            response.raise_for_status() # Lanza excepción para errores HTTP (4xx o 5xx)
            result = response.json()

            candidate = result.get('candidates', [{}])[0]
            if candidate and candidate.get('content', {}).get('parts', [{}])[0].get('text'):
                return candidate['content']['parts'][0]['text']
            else:
                # Si la API no devuelve contenido, usar una respuesta de fallback
                print(f"Respuesta inesperada de la API: {result}")
                return "Gracias por compartir eso. ¿Puedes contarme un poco más al respecto?"

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                print(f"Error de API, reintentando en {wait_time}s... ({e})")
                time.sleep(wait_time) # Espera antes de reintentar
            else:
                print(f"Error al conectar con la API de Gemini después de {MAX_RETRIES} intentos. {e}")
                return "Lo siento, estoy teniendo problemas para conectarme en este momento. Validemos lo que sientes. Es normal sentirse así. Por favor, intenta enviarme un mensaje de nuevo en un momento."
        except Exception as e:
            print(f"Error inesperado al procesar la respuesta de la API: {e}")
            return "Ocurrió un error inesperado al generar la respuesta. ¿Podrías reformularlo?"


# ---------------------------
# Utilidades: historial
# ---------------------------
def cargar_historial():
    if not os.path.exists(HISTORIAL_FILE):
        return []
    with open(HISTORIAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def guardar_en_historial(estado, texto):
    historial = cargar_historial()
    entry = {
        "fecha": datetime.datetime.now().isoformat(),
        "estado": estado,
        "texto": texto
    }
    historial.append(entry)
    with open(HISTORIAL_FILE, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)

# ---------------------------
# Clasificación / reglas terapéuticas
# ---------------------------
def clasificar_emocion(texto: str) -> str:
    try:
        result = sentiment_pipe(texto)[0]
        label = result["label"]
        # Este modelo (nlptown) SÍ usa estrellas, así que esta lógica es correcta
        if label in ["4 stars", "5 stars"]:
            return "positivo"
        elif label in ["3 stars"]:
            return "neutro"
        else: # 1 star, 2 stars
            return "negativo"
    except Exception:
        txt = texto.lower()
        if any(w in txt for w in ["feliz", "contento", "bien", "optimista"]):
            return "positivo"
        if any(w in txt for w in ["triste", "cansado", "vacío", "sin ganas"]):
            return "negativo"
        return "neutro"

# ---------------------------
# Motor terapéutico avanzado
# ---------------------------
def aplicar_reglas(texto: str, emocion_detectada: str, chat_history: list):
    txt = texto.lower()
    
    # Mantener el historial corto para la API
    history_context = chat_history[-6:]

    # === CRISIS (Respuesta estática e inmediata) ===
    crisis_patterns = [
        "me quiero morir", "ya no quiero seguir", "suicid",
        "lastimarme", "hacerme daño", "dañarme", "no quiero vivir"
    ]
    if any(p in txt for p in crisis_patterns):
        return "riesgo alto", (
            "Gracias por expresar algo tan importante. No estás solo.\n"
            "Estoy muy preocupado por ti. Si estás en peligro inmediato, por favor contacta a emergencias.\n"
            "Puedo darte líneas de ayuda y recursos si lo deseas. Estoy aquí contigo."
        )

    # === CBT (Llamada a Gemini con prompt de TCC) ===
    if any(p in txt for p in ["todo me sale mal", "nada me sale bien", "soy un fracaso", "siempre fallo", "nunca podré"]):
        system_prompt = (
            "Eres un terapeuta cognitivo-conductual (TCC). Tu respuesta debe ser concisa (máximo 3 oraciones). "
            "Ayuda al usuario a identificar la distorsión cognitiva (ej. generalización, todo-o-nada) en su último mensaje. "
            "Valida su emoción y luego haz una pregunta para desafiar ese pensamiento."
        )
        return "respuesta cbt", gemini_generate_response(system_prompt, history_context, texto)

    # === ACT (Llamada a Gemini con prompt de ACT) ===
    if any(p in txt for p in ["miedo", "fracasar", "fallar otra vez", "ansiedad por el futuro", "no puedo dejar de pensar en"]):
        system_prompt = (
            "Eres un terapeuta de Aceptación y Compromiso (ACT). Tu respuesta debe ser concisa (máximo 3 oraciones). "
            "No intentes cambiar el pensamiento. Valida la emoción, normalízala como parte de la experiencia humana (aceptación), "
            "y luego haz una pregunta sobre qué es importante para el usuario (valores) en esta situación."
        )
        return "respuesta act", gemini_generate_response(system_prompt, history_context, texto)

    # === DBT (Llamada a Gemini con prompt de DBT) ===
    if any(p in txt for p in ["abrumado", "saturado", "agotado", "agotada", "no puedo más", "demasiado intenso"]):
        system_prompt = (
            "Eres un terapeuta Dialéctico Conductual (DBT). Tu respuesta debe ser concisa (máximo 3 oraciones). "
            "Primero, valida intensamente la emoción ('Es completamente comprensible que te sientas así'). "
            "Luego, sugiere una habilidad de tolerancia al malestar muy breve y concreta (ej. 'intenta notar 3 cosas que puedas ver' o 'una respiración profunda')."
        )
        return "respuesta dbt", gemini_generate_response(system_prompt, history_context, texto)

    # === APOYO EMOCIONAL (Llamada a Gemini) ===
    if any(p in txt for p in ["nadie me entiende", "me siento solo", "me siento sola"]):
        system_prompt = (
            "Eres un compañero de apoyo emocional. Tu respuesta debe ser concisa (máximo 3 oraciones). "
            "Ofrece validación y conexión ('Estoy aquí contigo', 'Gracias por compartirlo'). "
            "Haz una pregunta abierta para invitarlo a compartir más sobre ese sentimiento de soledad."
        )
        return "apoyo emocional", gemini_generate_response(system_prompt, history_context, texto)

    # === NEUROCIENCIA (Llamada a Gemini) ===
    if any(p in txt for p in ["presión en el pecho", "no puedo relajarme", "tensión", "ansiedad física", "mi cuerpo está tenso"]):
        system_prompt = (
            "Eres un educador en neurociencia y mindfulness. Tu respuesta debe ser concisa (máximo 3 oraciones). "
            "Explica brevemente la conexión cuerpo-mente (ej. 'Esa es la respuesta de tu sistema nervioso al estrés') "
            "y sugiere una acción física simple (ej. 'intenta una exhalación larga para calmar ese sistema')."
        )
        return "respuesta neurociencia", gemini_generate_response(system_prompt, history_context, texto)

    # === TERAPIA EMOCIONAL MIXTA (¡La que causaba el problema!) ===
    # Ahora es dinámica y conversacional.
    if emocion_detectada == "negativo":
        system_prompt = (
            "Eres un chatbot terapéutico empático. Tu respuesta debe ser concisa (máximo 3 oraciones). "
            "Tu objetivo es NO repetirte. Mira el historial de chat."
            "Valida el sentimiento que el usuario acaba de expresar ('Entiendo que te sientas...'). "
            "Luego, haz una pregunta abierta y gentil para explorar el origen de esa emoción ('¿Qué crees que ha contribuido a eso?')."
        )
        return "terapia mixta", gemini_generate_response(system_prompt, history_context, texto)

    # === ESTADO POSITIVO (Dinámico) ===
    if emocion_detectada == "positivo":
        system_prompt = (
            "Eres un chatbot de psicología positiva. Tu respuesta debe ser concisa (máximo 3 oraciones). "
            "Celebra la emoción positiva del usuario ('¡Qué bueno leer eso!'). "
            "Luego, haz una pregunta para ayudarlo a anclar esa emoción ('¿Qué es lo que más estás disfrutando de eso?')."
        )
        return "estado positivo", gemini_generate_response(system_prompt, history_context, texto)

    # === ESTADO NEUTRO (Dinámico) ===
    system_prompt = (
        "Eres un chatbot terapéutico empático. Tu respuesta debe ser concisa (máximo 3 oraciones). "
        "El usuario ha dicho algo neutro. Simplemente acéptalo ('Entendido.', 'Gracias por compartirlo.') "
        "y haz una pregunta abierta para invitar a una exploración más profunda ('¿Hay algo más en tu mente hoy?')."
    )
    return "estado neutro", gemini_generate_response(system_prompt, history_context, texto)

# ---------------------------
# Seguimiento semanal
# ---------------------------
def filtrar_semana(historial):
    hoy = datetime.datetime.now()
    semana = hoy - datetime.timedelta(days=7)
    return [h for h in historial if datetime.datetime.fromisoformat(h["fecha"]) >= semana]

def generar_estadisticas_semana(historial_semana):
    if not historial_semana:
        return None

    df = pd.DataFrame(historial_semana)
    conteos = df["estado"].value_counts().to_dict()
    df["fecha_dt"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha_dt")

    # tendencia básica
    if len(df) >= 2:
        primeros = df.iloc[0]["estado"]
        ultimos = df.iloc[-1]["estado"]
        tendencia = "estable" if primeros == ultimos else f"cambio de '{primeros}' a '{ultimos}'"
    else:
        tendencia = "estable"

    return {
        "conteos": conteos,
        "tendencia": tendencia,
        "total_interacciones": len(df),
        "primer_registro": df.iloc[0]["fecha"],
        "ultimo_registro": df.iloc[-1]["fecha"]
    }

def generar_reporte_profesional(historial_semana, estadisticas):
    if not historial_semana:
        return "No hay suficiente información esta semana."

    df = pd.DataFrame(historial_semana)
    estados_frec = estadisticas["conteos"]
    principal = max(estados_frec, key=estados_frec.get)

    riesgo = df[df["estado"] == "riesgo alto"]
    hubo_riesgo = len(riesgo) > 0

    reporte = f"""
=== REPORTE SEMANAL PARA PROFESIONALES DE LA SALUD MENTAL ===

Periodo analizado:
  Desde: {estadisticas['primer_registro']}
  Hasta: {estadisticas['ultimo_registro']}

Total de interacciones:
  - {estadisticas['total_interacciones']}

Intervención predominante de la semana:
  - {principal.upper()} ({estados_frec[principal]} apariciones)

Distribución de intervenciones:
  {estados_frec}

Tendencia de intervención general:
  - {estadisticas['tendencia']}

Señales de riesgo detectadas:
  - {"Sí hubo señales de riesgo alto." if hubo_riesgo else "No se detectaron señales de riesgo crítico."}

Notas de observación:
  - Este análisis es automatizado y basado en texto. No constituye diagnóstico.
  - Se recomienda revisión profesional del contenido textual del usuario.

Últimas expresiones relevantes:
"""

    for idx, row in df.tail(5).iterrows():
        reporte += f"\n[{row['fecha']}] ({row['estado'].upper()}): {row['texto']}"

    reporte += "\n\n--- Fin del reporte ---\n"
    return reporte

# ---------------------------
# UI Streamlit
# ---------------------------
st.title("Chatbot Emocional Terapéutico")
st.caption("Herramienta educativa de apoyo emocional — No sustituye atención profesional.")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Conversación")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["sender"] == "user":
            st.markdown(f"**Tú:** {msg['text']}")
        else:
            st.markdown(f"**Chatbot ({msg['tag'].upper()}):** {msg['text']}")

    # Usar un formulario para que 'Enter' envíe el mensaje
    with st.form("chat_form", clear_on_submit=True):
        entrada = st.text_area("¿Cómo te sientes?", height=120, key="input_area")
        enviar = st.form_submit_button("Enviar")

    if enviar and entrada.strip():
        # Añadir mensaje del usuario al estado
        st.session_state.messages.append({"sender": "user", "text": entrada})
        
        # Mostrar spinner mientras se genera la respuesta
        with st.spinner("Pensando..."):
            emocion = clasificar_emocion(entrada)
            # Pasamos el historial de chat actual a la función de reglas
            tag, respuesta = aplicar_reglas(entrada, emocion, st.session_state.messages)
            guardar_en_historial(tag, entrada)

        st.session_state.messages.append({"sender": "bot", "text": respuesta, "tag": tag})
        
        # Usar st.rerun() que es la forma moderna
        st.rerun()

with col2:
    st.header("Estado General")
    historial = cargar_historial()

    if historial:
        last = historial[-1]
        st.write(f"**Último estado:** {last['estado'].upper()}")
        st.write(f"**Texto:** {last['texto']}")
    else:
        st.info("Aún no hay interacciones.")

    if st.button("Borrar historial"):
        if os.path.exists(HISTORIAL_FILE):
            os.remove(HISTORIAL_FILE)
        # Limpiar también el historial de sesión
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.subheader("Seguimiento semanal")

    historial_semana = filtrar_semana(historial)
    estadisticas = generar_estadisticas_semana(historial_semana)

    if estadisticas:
        st.write("**Interacciones esta semana:**", estadisticas["total_interacciones"])
        st.write("**Tendencia emocional:**", estadisticas["tendencia"])
        
        # Renombrar 'conteo' para el gráfico
        df_conteos = pd.DataFrame(estadisticas["conteos"].items(), columns=["Intervención", "Conteo"])
        df_conteos = df_conteos.set_index("Intervención")
        st.bar_chart(df_conteos)

        st.subheader("Reporte para profesionales")
        reporte = generar_reporte_profesional(historial_semana, estadisticas)
        st.text(reporte)
    else:
        st.info("Aún no hay suficientes datos esta semana.")