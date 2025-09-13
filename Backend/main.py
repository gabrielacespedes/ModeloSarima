from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from pmdarima import auto_arima

app = FastAPI(title="API de Predicción de Ventas")

@app.get("/predict")
def predict(horizon: int = Query(14, description="Número de días a predecir")):
    try:
        # ===========================
        # Leer datos
        # ===========================
        df = pd.read_excel("ventas_raw.xlsx")
        if "Fecha Emisión" not in df.columns or "Importe Final" not in df.columns:
            return {"error": "El archivo debe tener 'Fecha Emisión' y 'Importe Final'"}

        df["Fecha Emisión"] = pd.to_datetime(df["Fecha Emisión"])
        df = df.groupby("Fecha Emisión", as_index=False)["Importe Final"].sum()
        df = df.rename(columns={"Fecha Emisión": "ds", "Importe Final": "y"})

        # ===========================
        # Serie diaria completa
        # ===========================
        full_range = pd.date_range(df["ds"].min(), df["ds"].max(), freq="D")
        df = df.set_index("ds").reindex(full_range).rename_axis("ds").reset_index()
        df["y"] = df["y"].replace(0, np.nan)
        df["y"] = df["y"].fillna(df["y"].rolling(7, min_periods=1).mean())
        df["y"] = df["y"].fillna(method="bfill").fillna(method="ffill")

        # ===========================
        # Modelo SARIMA Auto
        # ===========================
        model = auto_arima(df["y"], seasonal=True, m=7, suppress_warnings=True)
        forecast = model.predict(n_periods=horizon, return_conf_int=True)
        fechas_forecast = pd.date_range(df["ds"].max() + pd.Timedelta(days=1), periods=horizon)

        forecast_df = pd.DataFrame({
            "Fecha": fechas_forecast.astype(str),
            "Predicción": forecast[0],
            "LI": forecast[1][:,0],
            "LS": forecast[1][:,1]
        })

        historico_df = df.copy()
        historico_df["ds"] = historico_df["ds"].astype(str)
        historico_df.rename(columns={"ds": "Fecha", "y": "Ventas"}, inplace=True)

        return JSONResponse(content={
            "historico": historico_df.to_dict(orient="records"),
            "forecast": forecast_df.to_dict(orient="records")
        })

    except Exception as e:
        return {"error": str(e)}
