# app.py
import streamlit as st
import json
import pandas as pd
import altair as alt
from datetime import datetime, timezone
from agent import ejecutar_agente
from model_utils import MODELOS

st.set_page_config(page_title="ExplicaMe", page_icon="🧠", layout="wide")

# --- Estilos profesionales e inclusivos ---
st.markdown("""
<style>
.big-title {font-size: 2.2rem; font-weight: 700; margin-bottom: 0; color: #1f2937;}
.subtitle {color: #4b5563; margin-top: 0.2rem; font-size: 1rem;}
.card {background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem 1.2rem;}
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 8px;
    font-size: 0.8rem; font-weight: 600; margin-right: 4px;
    background: #eef2ff; color: #3730a3;
}
.badge-green {background: #dcfce7; color: #166534;}
.badge-red {background: #fee2e2; color: #991b1b;}
.badge-gray {background: #f3f4f6; color: #374151;}
.section-title {font-size: 1.1rem; font-weight: 600; color: #111827; margin-top: 0.5rem;}
/* Mejora de contraste para accesibilidad */
[data-testid="stMetricValue"] {color: #111827;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🧠 ExplicaMe</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Agente adaptativo de explicaciones · Explicabilidad y transparencia para decisiones de crédito</p>',
    unsafe_allow_html=True,
)

# --- Sidebar ---
st.sidebar.header("📋 Datos de la solicitud")
ingreso = st.sidebar.number_input(
    "Ingreso mensual ($)", min_value=1200000, max_value=2500000, value=1200000, step=50000
)
deuda = st.sidebar.number_input(
    "Deuda actual ($)", min_value=0, max_value=150000, value=150000, step=5000
)
antiguedad = st.sidebar.number_input(
    "Antigüedad laboral (años)", min_value=0, max_value=40, value=33
)
historial = st.sidebar.selectbox(
    "Historial de pago", [0, 1], format_func=lambda x: "Bueno" if x == 1 else "Con impagos"
)

st.sidebar.divider()
st.sidebar.header("⚙️ Configuración")
nombre_modelo = st.sidebar.selectbox("Modelo de scoring", list(MODELOS.keys()))
audiencia = st.sidebar.selectbox(
    "🎯 Audiencia", ["Dirección", "Negocio", "Técnico", "Persona afectada"]
)

solicitud = {
    "ingreso_mensual": ingreso,
    "deuda_actual": deuda,
    "antiguedad_laboral": antiguedad,
    "historial_pago": historial,
}

# --- Botones de acción ---
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    generar = st.button("Generar explicación adaptada", type="primary", use_container_width=True)
with col_btn2:
    comparar = st.button("🔀 Comparar las 4 audiencias", use_container_width=True)

if generar:
    with st.spinner("El agente está analizando la solicitud..."):
        try:
            st.session_state["log_entry"] = ejecutar_agente(solicitud, audiencia, nombre_modelo)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

if comparar:
    with st.spinner("Generando las 4 explicaciones (esto tarda unos segundos)..."):
        try:
            audiencias = ["Dirección", "Negocio", "Técnico", "Persona afectada"]
            st.session_state["comparacion"] = {
                aud: ejecutar_agente(solicitud, aud, nombre_modelo) for aud in audiencias
            }
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

# =========================================================
# RESULTADO INDIVIDUAL
# =========================================================
if "log_entry" in st.session_state:
    entry = st.session_state["log_entry"]

    # Badges de transparencia
    decision_badge = "badge-green" if entry["decision"] == "APROBADO" else "badge-red"
    st.markdown(
        f'<span class="badge {decision_badge}">Decisión: {entry["decision"]}</span> '
        f'<span class="badge">Modelo: {entry["modelo"]}</span> '
        f'<span class="badge badge-gray">{entry["modelo_tipo"]}</span> '
        f'<span class="badge badge-gray">Versión {entry["version_modelo"]}</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    col1, col2, col3 = st.columns(3)
    col1.metric("Decisión", entry["decision"])
    col2.metric("Probabilidad de aprobación", f"{entry['probabilidad']:.2%}")
    col3.metric("Audiencia", entry["audiencia"])

    st.subheader(f"Explicación para: {entry['audiencia']}")
    st.info(entry["explicacion_generada"])

    # --- Validación de fidelidad ---
    with st.expander("🔍 Validación de fidelidad"):
        v = entry["validacion"]
        st.write(f"Variables del modelo: `{', '.join(v['variables_reales'])}`")
        st.write(f"Variables mencionadas: `{v['variables_mencionadas'] or 'ninguna'}`")
        if v["fidelidad_ok"]:
            st.success("✅ La explicación menciona al menos una variable real del modelo.")
        else:
            st.warning(f"⚠️ {v['advertencia']}")

    # --- SHAP: gráfica de barras + tabla ---
    with st.expander("📊 Contribuciones SHAP (evidencia técnica)", expanded=True):
        df_shap = pd.DataFrame(
            [{"variable": k, "contribucion": v} for k, v in entry["contribuciones_shap"].items()]
        )
        df_shap["impacto"] = df_shap["contribucion"].abs()
        df_shap = df_shap.sort_values("impacto", ascending=False)

        chart = (
            alt.Chart(df_shap)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                x=alt.X(
                    "contribucion:Q",
                    title="Contribución SHAP (↑ aprueba / ↓ rechaza)",
                ),
                y=alt.Y("variable:N", sort="-x", title=None),
                color=alt.condition(
                    alt.datum.contribucion > 0,
                    alt.value("#16a34a"),   # verde accesible
                    alt.value("#dc2626"),   # rojo accesible
                ),
                tooltip=[
                    alt.Tooltip("variable:N", title="Variable"),
                    alt.Tooltip("contribucion:Q", format="+.4f", title="Contribución"),
                ],
            )
            .properties(height=220, title="¿Qué variables pesaron en la decisión?")
        )
        st.altair_chart(chart, use_container_width=True)

        st.dataframe(
            df_shap[["variable", "contribucion"]]
            .rename(columns={"variable": "Variable", "contribucion": "Contribución SHAP"})
            .style.format({"Contribución SHAP": "{:+.4f}"}),
            hide_index=True,
            use_container_width=True,
        )

    # --- Transparencia / fuentes ---
    with st.expander("🧾 Trazabilidad y fuentes"):
        st.write(f"**Fuentes:** {', '.join(entry['fuentes'])}")
        st.write(f"**Versión del modelo:** {entry['version_modelo']}")
        st.write(f"**Hash de evidencia:** `{entry['hash_evidencia']}`")
        st.write(f"**Timestamp (UTC):** {entry['timestamp']}")
        st.write(f"**LLM:** {entry['modelo_llm']}")

    # --- Canal de revisión: feedback + apelación ---
    st.subheader("🗣️ Canal de revisión")
    st.caption("Tu feedback queda registrado en el log auditable.")

    col_a, col_b = st.columns(2)
    with col_a:
        fb = st.radio(
            "¿La explicación fue útil?", ["👍 Sí", "👎 No"], horizontal=True, key="fb"
        )
        motivo = ""
        if fb == "👎 No":
            motivo = st.text_area(
                "¿Por qué no fue útil?", placeholder="Explica brevemente...", key="motivo"
            )
        if st.button("Enviar feedback", key="btn_fb"):
            st.session_state["log_entry"]["feedback"] = {"util": fb, "motivo": motivo}
            st.success("Feedback registrado en el log.")

    with col_b:
        st.write("¿Quieres solicitar una revisión de la decisión?")
        if st.button("Solicitar apelación", key="btn_ap"):
            st.session_state["log_entry"]["apelacion"] = {
                "solicitada": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            st.info(
                "Apelación registrada. Un revisor humano contactará a la persona solicitante."
            )

    # --- Descarga del log ---
    st.divider()
    json_str = json.dumps(st.session_state["log_entry"], indent=2, ensure_ascii=False)
    st.download_button(
        label="📥 Descargar log auditable (JSON)",
        data=json_str,
        file_name=f"log_explicame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )

# =========================================================
# COMPARACIÓN DE 4 AUDIENCIAS + HEATMAP
# =========================================================
if "comparacion" in st.session_state:
    st.divider()
    st.subheader("🔀 Comparación de las 4 audiencias")
    st.caption("Misma decisión y misma evidencia SHAP. Cambia la explicación según la audiencia.")

    comp = st.session_state["comparacion"]
    primera = next(iter(comp.values()))

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Decisión", primera["decision"])
    col_b.metric("Probabilidad", f"{primera['probabilidad']:.2%}")
    col_c.metric("Modelo", primera["modelo"])

    # --- Explicaciones lado a lado ---
    cols = st.columns(4)
    for col, (aud, data) in zip(cols, comp.items()):
        with col:
            fidelidad_badge = (
                '<span class="badge badge-green">✅ fiel</span>'
                if data["validacion"]["fidelidad_ok"]
                else '<span class="badge badge-red">⚠️ revisar</span>'
            )
            st.markdown(f"**{aud}** {fidelidad_badge}", unsafe_allow_html=True)
            st.info(data["explicacion_generada"])

    # --- HEATMAP variables × audiencias ---
    st.subheader("🔥 Heatmap: variables mencionadas por audiencia")
    st.caption(
        "Verde = la explicación menciona la variable · Gris = no la menciona. "
        "Permite ver de un vistazo qué evidencia destaca cada perfil."
    )

    variables = list(primera["contribuciones_shap"].keys())
    audiencias = list(comp.keys())

    # Construir matriz 0/1 de menciones
    matriz = []
    for aud in audiencias:
        mencionadas = [m.lower().replace("_", " ") for m in comp[aud]["validacion"]["variables_mencionadas"]]
        for var in variables:
            var_norm = var.lower().replace("_", " ")
            matriz.append(
                {
                    "Audiencia": aud,
                    "Variable": var,
                    "Mencionada": 1 if var_norm in mencionadas else 0,
                }
            )
    df_heat = pd.DataFrame(matriz)

    heatmap = (
        alt.Chart(df_heat)
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X("Audiencia:N", title=None, sort=audiencias),
            y=alt.Y("Variable:N", title=None, sort=variables),
            color=alt.condition(
                alt.datum.Mencionada == 1,
                alt.value("#22c55e"),   # verde accesible
                alt.value("#e5e7eb"),   # gris claro
            ),
            tooltip=[
                alt.Tooltip("Audiencia:N"),
                alt.Tooltip("Variable:N"),
                alt.Tooltip("Mencionada:N", title="Mencionada (1=sí)"),
            ],
        )
        .properties(height=260, title="Variables del modelo mencionadas en cada explicación")
    )
    st.altair_chart(heatmap, use_container_width=True)

    # --- Tabla resumen de fidelidad ---
    with st.expander("📋 Resumen de fidelidad por audiencia"):
        resumen = pd.DataFrame(
            [
                {
                    "Audiencia": aud,
                    "Fidelidad OK": "✅" if data["validacion"]["fidelidad_ok"] else "⚠️",
                    "Variables mencionadas": ", ".join(data["validacion"]["variables_mencionadas"]) or "ninguna",
                }
                for aud, data in comp.items()
            ]
        )
        st.dataframe(resumen, hide_index=True, use_container_width=True)

    # --- Exportar comparación ---
    comp_json = json.dumps(comp, indent=2, ensure_ascii=False)
    st.download_button(
        label="📥 Descargar comparación completa (JSON)",
        data=comp_json,
        file_name=f"comparacion_audiencias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )

st.divider()
st.caption(
    "🔒 Transparencia: cada paso queda registrado. La explicación se valida contra la evidencia SHAP. "
    "El feedback y las apelaciones se incorporan al log auditable."
)