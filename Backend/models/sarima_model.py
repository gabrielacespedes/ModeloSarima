import pandas as pd
from pmdarima import auto_arima

def entrenar_y_predecir(df: pd.DataFrame, periods: int = 14, m: int = 14):
    model = auto_arima(df["y"], seasonal=True, m=m, suppress_warnings=True)
    forecast = model.predict(n_periods=periods)
    fechas_forecast = pd.date_range(df["ds"].max() + pd.Timedelta(days=1), periods=periods)
    return pd.DataFrame({"Fecha": fechas_forecast, "Predicción": forecast})
