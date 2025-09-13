import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# ==============================
# CONFIGURACIÓN DE LA APP
# ==============================
st.set_page_config(page_title="Predicción de Ventas BI", page_icon="📊", layout="wide")
st.title("📊 Sistema de Predicción de Ventas con SARIMA")
st.markdown("### Bienvenido, carga tus datos y obtén predicciones y análisis detallados.")

# ==============================
# CARGA DE DATOS
# ==============================
uploaded_file = st.file_uploader("📂 Sube archivo de ventas (Excel)", type=["xlsx"])
with st.spinner("Cargando datos..."):
    if uploaded_file:
        df_hist = pd.read_excel(uploaded_file)
    else:
        df_hist = pd.read_excel("ventas_raw.xlsx")


df_hist = df_hist[["Fecha Emisión", "Importe Final", "Doc. Auxiliar", "Razón Social"]].copy()
df_hist["Fecha Emisión"] = pd.to_datetime(df_hist["Fecha Emisión"])

df_sum = df_hist.groupby("Fecha Emisión", as_index=False)["Importe Final"].sum()
full_range = pd.date_range(df_sum["Fecha Emisión"].min(), df_sum["Fecha Emisión"].max(), freq="D")
df_sum = df_sum.set_index("Fecha Emisión").reindex(full_range).fillna(0).rename_axis("Fecha").reset_index()

df_sum["Importe Final"] = df_sum["Importe Final"].replace(0, np.nan)
df_sum["Importe Final"] = df_sum["Importe Final"].fillna(df_sum["Importe Final"].rolling(7, min_periods=1).mean())
df_sum["Importe Final"] = df_sum["Importe Final"].fillna(method="bfill").fillna(method="ffill")

# ==============================
# SLIDER HORIZONTE
# ==============================
horizon = st.slider("Selecciona horizonte de predicción (días):", min_value=7, max_value=14, value=14)

# ==============================
# ENTRENAR SARIMAX
# ==============================
@st.cache_resource
def entrenar_sarima(series, seasonal_period=7):
    model = SARIMAX(series,
                    order=(1,1,1),
                    seasonal_order=(1,1,1,seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False)
    return results

with st.spinner("Entrenando modelo..."):
    modelo = entrenar_sarima(df_sum["Importe Final"], seasonal_period=14)
    forecast = modelo.forecast(steps=horizon)
    fechas_forecast = pd.date_range(df_sum["Fecha"].max() + pd.Timedelta(days=1), periods=horizon)
    df_forecast = pd.DataFrame({"Fecha": fechas_forecast, "Predicción": forecast})

# ==============================
# PESTAÑAS
# ==============================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Dashboard",
    "📊 Tabla de Predicciones",
    "📉 Evaluación del Modelo",
    "👥 Análisis por Clientes",
    "📆 Estacionalidad y Tendencias"
])

# ------------------------------
# TAB 1: Dashboard
# ------------------------------
with tab1:
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df_sum["Fecha"], df_sum["Importe Final"], label="Histórico", marker="o")
    ax.plot(df_forecast["Fecha"], df_forecast["Predicción"], label="Predicción", marker="x", color="red")
    ax.set_title("Ventas Histórico + Predicción")
    ax.set_ylabel("Ventas (S/)")
    ax.legend()
    st.pyplot(fig)

# ------------------------------
# TAB 2: Tabla de Predicciones
# ------------------------------
with tab2:
    st.dataframe(df_forecast[["Fecha", "Predicción"]], use_container_width=True)
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_forecast[["Fecha", "Predicción"]].to_excel(writer, index=False, sheet_name="Predicciones")
    st.download_button("⬇️ Descargar predicciones", data=output.getvalue(),
                       file_name="predicciones_sarima.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ------------------------------
# TAB 3: Evaluación del Modelo
# ------------------------------
with tab3:
    df_eval = df_sum.copy()
    df_eval["yhat"] = modelo.fittedvalues
    df_eval_recent = df_eval.tail(horizon)
    df_eval_recent["error"] = df_eval_recent["Importe Final"] - df_eval_recent["yhat"]

    rmse = mean_squared_error(df_eval_recent["Importe Final"], df_eval_recent["yhat"])**0.5
    mape = (abs(df_eval_recent["error"]/df_eval_recent["Importe Final"])).mean()*100

    col1, col2 = st.columns(2)
    col1.metric("📏 RMSE (últimos días)", f"{rmse:.2f}")
    col2.metric("📐 MAPE (últimos días)", f"{mape:.2f} %")

    fig_eval, ax_eval = plt.subplots(figsize=(10,4))
    ax_eval.plot(df_eval_recent["Fecha"], df_eval_recent["Importe Final"], label="Real", marker="o")
    ax_eval.plot(df_eval_recent["Fecha"], df_eval_recent["yhat"], label="Predicción", marker="x")
    ax_eval.legend()
    st.pyplot(fig_eval)

# ------------------------------
# TAB 4: Análisis por Clientes
# ------------------------------
with tab4:
    st.subheader("📊 Análisis de Clientes (BI)")
    total_ventas = df_hist.groupby("Razón Social")["Importe Final"].sum().sum()
    num_clientes = df_hist["Razón Social"].nunique()
    ticket_promedio = df_hist["Importe Final"].sum() / df_hist.shape[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Ventas", f"S/ {total_ventas:,.0f}")
    col2.metric("👥 Número de Clientes", f"{num_clientes}")
    col3.metric("🧾 Ticket Promedio", f"S/ {ticket_promedio:,.2f}")

    top_clientes = df_hist.groupby(["Doc. Auxiliar", "Razón Social"])["Importe Final"].sum().sort_values(ascending=False).head(10).reset_index()
    st.markdown("### 🏆 Top 10 Clientes por Ventas")
    st.dataframe(top_clientes, use_container_width=True, height=300)

    cliente_seleccionado = st.selectbox("Selecciona un cliente para ver su evolución", top_clientes["Razón Social"].unique())
    df_cliente = df_hist[df_hist["Razón Social"] == cliente_seleccionado].sort_values("Fecha Emisión")

    st.markdown(f"### 📈 Evolución de ventas: {cliente_seleccionado}")
    fig_cliente, ax_cliente = plt.subplots(figsize=(10,4))
    ax_cliente.plot(df_cliente["Fecha Emisión"], df_cliente["Importe Final"], marker="o", color="tab:blue")
    ax_cliente.set_xlabel("Fecha")
    ax_cliente.set_ylabel("Ventas (S/)")
    ax_cliente.set_title(f"Ventas Diarias de {cliente_seleccionado}")
    st.pyplot(fig_cliente)

    df_cliente["mes"] = df_cliente["Fecha Emisión"].dt.month
    ventas_mes = df_cliente.groupby("mes")["Importe Final"].sum().reset_index()
    st.markdown(f"### 📅 Ventas por Mes: {cliente_seleccionado}")
    fig_mes, ax_mes = plt.subplots(figsize=(10,4))
    ax_mes.bar(ventas_mes["mes"], ventas_mes["Importe Final"], color="tab:orange")
    ax_mes.set_xlabel("Mes")
    ax_mes.set_ylabel("Ventas (S/)")
    ax_mes.set_title(f"Estacionalidad Mensual de {cliente_seleccionado}")
    st.pyplot(fig_mes)

    st.markdown("### 📊 Distribución de Ventas por Cliente")
    ventas_clientes = df_hist.groupby("Razón Social")["Importe Final"].sum().sort_values(ascending=False).reset_index()
    fig_dist, ax_dist = plt.subplots(figsize=(10,4))
    ax_dist.barh(ventas_clientes["Razón Social"].head(20), ventas_clientes["Importe Final"].head(20), color="tab:green")
    ax_dist.set_xlabel("Ventas (S/)")
    ax_dist.set_ylabel("Clientes")
    ax_dist.set_title("Top 20 Clientes")
    st.pyplot(fig_dist)

# ------------------------------
# TAB 5: Estacionalidad y Tendencias
# ------------------------------
with tab5:
    st.subheader("📆 Tendencias Estacionales")

    # Promedio por semana del año
    df_sum["Semana"] = df_sum["Fecha"].dt.isocalendar().week
    weekly_avg = df_sum.groupby("Semana")["Importe Final"].mean().reset_index()

    fig_season, ax_season = plt.subplots(figsize=(10,5))
    ax_season.plot(weekly_avg["Semana"], weekly_avg["Importe Final"], marker="o", color="green")
    ax_season.set_title("Promedio de ventas por semana (tendencia estacional)")
    ax_season.set_xlabel("Semana del año")
    ax_season.set_ylabel("Ventas promedio (S/)")
    st.pyplot(fig_season)

    # Promedio por día de la semana
    df_sum["DiaSemana"] = df_sum["Fecha"].dt.day_name()
    weekday_avg = df_sum.groupby("DiaSemana")["Importe Final"].mean().reindex([
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]).reset_index()

    fig_weekday, ax_weekday = plt.subplots(figsize=(10,5))
    ax_weekday.bar(weekday_avg["DiaSemana"], weekday_avg["Importe Final"], color="skyblue")
    ax_weekday.set_title("Promedio de ventas por día de la semana")
    ax_weekday.set_xlabel("Día")
    ax_weekday.set_ylabel("Ventas promedio (S/)")
    st.pyplot(fig_weekday)
