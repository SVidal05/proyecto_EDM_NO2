import streamlit as st
import joblib
import plotly.graph_objects as go

# --- Configuración de la página (debe ir lo primero) ---
st.set_page_config(page_title="Alerta NO2 Valencia", page_icon="🌫️", layout="wide")

# --- Cargar modelo (cacheado para que no recargue en cada interacción) ---
@st.cache_resource
def cargar_modelo():
    datos = joblib.load("modelo_no2.joblib")
    return datos["modelo"], datos["features"], datos["metricas"]

modelo, features, metricas = cargar_modelo()

# --- Cabecera ---
st.title("🌫️ Sistema de alerta de calidad del aire de Valencia")
st.caption("Predicción de alertas de NO₂ · Estación Avda. Francia · Datos: Red de Vigilancia (2016–2021)")

# --- Pestañas ---
tab_pred, tab_mon, tab_info = st.tabs(["🔮 Predicción", "📊 Monitorización (drift)", "ℹ️ Sobre el modelo"])

with tab_pred:
    st.subheader("Introduce la situación actual")
    col1, col2 = st.columns(2)
    with col1:
        no2_ant = st.slider("NO₂ hace una hora (µg/m³)", 0, 200, 40)
        hora    = st.slider("Hora del día", 0, 23, 8)
        viento  = st.slider("Velocidad del viento (m/s)", 0.0, 15.0, 2.0)
    with col2:
        temp    = st.slider("Temperatura (°C)", -5.0, 45.0, 20.0)
        humedad = st.slider("Humedad relativa (%)", 0, 100, 60)
        precip  = st.slider("Precipitación (mm)", 0.0, 50.0, 0.0)

    entrada = [[no2_ant, hora, viento, temp, humedad, precip]]
    prob = modelo.predict_proba(entrada)[0][1]

    st.divider()
    res1, res2 = st.columns([1, 1])

    with res1:
        # Medidor visual (gauge) con la probabilidad
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": "Probabilidad de alerta"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#d62728" if prob >= 0.5 else "#2ca02c"},
                "steps": [
                    {"range": [0, 50],  "color": "#e8f5e9"},
                    {"range": [50, 100], "color": "#ffebee"},
                ],
                "threshold": {"line": {"color": "black", "width": 3},
                              "thickness": 0.75, "value": 50},
            },
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with res2:
        st.write("")  # pequeño espacio
        st.write("")
        if prob >= 0.5:
            st.error(f"### ⚠️ ALERTA\nRiesgo alto de superar el umbral de NO₂ (40 µg/m³).")
        else:
            st.success(f"### ✅ Sin alerta\nNo se prevé superar el umbral de NO₂.")
        st.metric("Riesgo estimado", f"{prob*100:.0f}%")

with tab_mon:
    st.subheader("Monitorización de Data Drift")
    st.write(
        "El modelo se entrenó con datos hasta **septiembre de 2020**. "
        "Aquí comprobamos si la distribución del NO₂ ha **cambiado** con el tiempo "
        "(*data drift* / *covariate shift*), lo que degradaría las predicciones."
    )

    import pandas as pd
    df_drift = pd.read_csv("datos_drift.csv", parse_dates=["FechaHora"])
    df_drift["año"] = df_drift["FechaHora"].dt.year

    # Periodo de referencia (entrenamiento) vs periodo reciente
    referencia = df_drift[df_drift["año"] <= 2019]["NO2"]
    reciente   = df_drift[df_drift["año"] >= 2020]["NO2"]

    c1, c2, c3 = st.columns(3)
    c1.metric("NO₂ medio 2016–2019", f"{referencia.mean():.1f} µg/m³")
    c2.metric("NO₂ medio 2020–2021", f"{reciente.mean():.1f} µg/m³",
              delta=f"{reciente.mean() - referencia.mean():.1f}")
    caida = (1 - reciente.mean() / referencia.mean()) * 100
    c3.metric("Caída", f"{caida:.0f}%")

    # Histograma comparando las dos distribuciones
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=referencia, name="2016–2019 (entrenamiento)",
                               opacity=0.6, histnorm="probability density"))
    fig.add_trace(go.Histogram(x=reciente, name="2020–2021 (reciente)",
                               opacity=0.6, histnorm="probability density"))
    fig.update_layout(barmode="overlay", height=400,
                      xaxis_title="NO₂ (µg/m³)", yaxis_title="Densidad",
                      title="Distribución del NO₂: ¿ha cambiado?")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Interpretación:** si las dos distribuciones no coinciden, hay *data drift*. "
        "El desplome del NO₂ en 2020 (confinamiento + menos tráfico) es un caso real de "
        "*covariate shift*: el modelo ve datos distintos a los de su entrenamiento, "
        "lo que justifica **reentrenarlo periódicamente** en un sistema en producción."
    )


with tab_info:
    st.subheader("¿Cómo funciona?")
    st.write(
        "Este sistema predice si la concentración de **NO₂** superará el umbral de "
        "**40 µg/m³** en la próxima medición, usando un modelo **Random Forest** "
        "entrenado con datos horarios de la estación de Avda. Francia (2016–2021)."
    )

    st.markdown("#### 📊 Rendimiento del modelo (sobre datos de test no vistos)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC", metricas["AUC"])
    m2.metric("F1 (alerta)", metricas["F1 (alerta)"])
    m3.metric("Recall (alerta)", metricas["Recall (alerta)"])
    m4.metric("Precision (alerta)", metricas["Precision (alerta)"])
    st.caption(
        "Evaluación con separación temporal (entrenar con el pasado, validar con el futuro) "
        "para evitar fuga de información. Métricas centradas en la clase 'alerta' por el "
        "desbalanceo de clases (~82% sin alerta)."
    )

    st.markdown("#### 🔎 Pistas (features) que usa el modelo")
    explicacion = {
        "NO2_anterior": "NO₂ de la hora anterior — la pista más potente (la contaminación tiene inercia).",
        "hora_num": "Hora del día (0–23) — capta los picos de hora punta.",
        "Velocidad del viento": "Viento — dispersa la contaminación: más viento, menos NO₂.",
        "Temperatura": "Temperatura ambiente.",
        "Humedad relativa": "Humedad relativa del aire.",
        "Precipitación": "Lluvia — también ayuda a limpiar el aire.",
    }
    for f in features:
        st.markdown(f"- **{f}**: {explicacion.get(f, '')}")

    st.caption(
        "Nota: no se usan NO ni NOx como pistas porque se miden en el mismo instante "
        "que el NO₂ y no estarían disponibles al predecir (evitamos *data leakage*)."
    )