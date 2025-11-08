# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 10:45:50 2025

@author: jefte
"""

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
import math
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# =============================
# CONFIGURACIÓN GLOBAL
# =============================
symbol = "BTCUSDT"
interval = "4h"
days = 8 * 360
limit = 1000
CHECK_INTERVAL = 60 * 60 * 4  # cada 10 minutos revisa (ajusta según tu necesidad)

# --- Parámetros de estrategia ---
factor_riesgo = 0.025          # 2.5%
RISK_REWARD_RATIO = 1          # 1:1
factor_venta_breack = 0.5      # vender 50% al TP1
factor_gan_max = 3 * factor_riesgo  # TP final 7.5%

# Configuración WhatsApp GoWA
GOWA_URL = "http://localhost:3000/send-message"
WHATSAPP_NUMBER = "573163676816@c.us"  # <== cambia por tu número con código país

ENABLE_LONGS = True
ENABLE_SHORTS = True

# =============================
# FUNCIONES AUXILIARES
# =============================
def get_binance_klines(symbol, interval, start, end):
    url = "https://api.binance.com/api/v3/klines"
    df_all = []
    start_time = int(start.timestamp() * 1000)
    end_time = int(end.timestamp() * 1000)

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time
        }
        data = requests.get(url, params=params).json()
        if not data:
            break
        df_chunk = pd.DataFrame(data, columns=[
            "timestamp", "Open", "High", "Low", "Close", "Volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df_all.append(df_chunk)
        last_time = data[-1][0]
        if last_time >= end_time or len(data) < limit:
            break
        start_time = last_time + 1

    df = pd.concat(df_all, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df = df.set_index("timestamp")
    return df

# --- Configuración del correo ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USER = "jeftedanielp@gmail.com"
EMAIL_PASS = "jdbt xqws ortc ifrl"  # no la normal

def send_email(subject, body, to="dfgchnico@gmail.com"):
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_USER
        msg["To"] = to
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)

        print(f"✅ Correo enviado a {to}: {subject}")

    except Exception as e:
        print(f"❌ Error al enviar correo: {e}")

def detect_support_resistance(df, end, merge_threshold=0.08, months_back=6, min_touches=3):
    df = df.copy()
    cutoff_date = end - timedelta(days=months_back * 30)
    df_recent = df[df.index >= cutoff_date].copy()
    if df_recent.empty:
        return pd.DataFrame()

    highs = df_recent["High"].values
    lows = df_recent["Low"].values
    indices = np.arange(len(df_recent))

    adaptive_order = max(3, len(df_recent) // 50)
    low_idx = argrelextrema(lows, np.less_equal, order=adaptive_order)[0]
    high_idx = argrelextrema(highs, np.greater_equal, order=adaptive_order)[0]

    supports = lows[low_idx]
    resistances = highs[high_idx]

    def merge_levels(levels):
        if len(levels) == 0:
            return []
        levels = np.sort(levels)
        merged = []
        cluster = [levels[0]]
        for level in levels[1:]:
            if abs(level - cluster[-1]) / cluster[-1] <= merge_threshold:
                cluster.append(level)
            else:
                merged.append(np.mean(cluster))
                cluster = [level]
        merged.append(np.mean(cluster))
        return merged

    supports = merge_levels(supports)
    resistances = merge_levels(resistances)
    levels_data = []

    for s in supports:
        levels_data.append({"type": "support", "level": s})
    for r in resistances:
        levels_data.append({"type": "resistance", "level": r})

    return pd.DataFrame(levels_data)

# =============================
# DETECCIÓN DE SEÑALES
# =============================
def check_signals():
    end = datetime.utcnow()
    start = end - timedelta(days=days)

    # Cargar datos
    df_4h = get_binance_klines(symbol, "4h", start, end)
    df_semanal = get_binance_klines(symbol, "1w", start, end)

    # SMA semanal
    df_semanal["SMA_50"] = df_semanal["Close"].rolling(50).mean()
    supports_semanal = detect_support_resistance(df_semanal, end, merge_threshold=0.02, months_back=6)
    weekly_levels = supports_semanal['level'].dropna().tolist()
    weekly_resistances = supports_semanal['level'].dropna().tolist()

    # Mapeo SMA
    df_4h["SMA_50_weekly"] = df_4h.index.map(
        lambda ts: df_semanal.loc[df_semanal.index <= ts, "SMA_50"].iloc[-1]
        if (df_semanal.index <= ts).any() else np.nan
    )

    # Volumen Z-Score
    n = 20
    z_threshold = 2
    df_4h["vol_mean"] = df_4h["Volume"].rolling(window=n).mean()
    df_4h["vol_std"] = df_4h["Volume"].rolling(window=n).std()
    df_4h["vol_zscore"] = (df_4h["Volume"] - df_4h["vol_mean"]) / df_4h["vol_std"]
    df_4h["HighVolume"] = df_4h["vol_zscore"] > z_threshold

    # Señales LONG
    df_4h["Signal_Long"] = False
    df_4h["Signal_Short"] = False

    if len(df_4h) < 2:
        return None
    
    prev2, prev, curr = df_4h.iloc[-3],df_4h.iloc[-2], df_4h.iloc[-1]
        
    # --- Long ---
    if ENABLE_LONGS and weekly_levels:
        nearest_support = min(weekly_levels, key=lambda x: abs(prev["Close"] - x))
        if (prev2["Low"] <= nearest_support <= prev2["High"]) and prev["Close"] > nearest_support and prev2["HighVolume"]:
            entry = curr["Open"]
            sl = entry * (1 - factor_riesgo)
            tp1 = entry * (1 + factor_riesgo)
            tp2 = entry * (1 + factor_gan_max)
            msg = f"🟢 Señal LONG detectada en {symbol}\n" \
                  f"💰 Entrada: {entry:.2f}\n" \
                  f"🛑 Stop Loss: {sl:.2f}\n" \
                  f"🎯 TP1 (1R): {tp1:.2f}\n" \
                  f"🎯 TP2 (3R): {tp2:.2f}\n" \
                  f"📊 Parcial: {factor_venta_breack*100:.0f}% en TP1\n" \
                  f"📆 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
            send_email(msg)
            return "LONG"
    
    # --- Short ---
    if ENABLE_SHORTS and weekly_resistances:
        resistances_above = [x for x in weekly_resistances if x > prev2["Open"]]
        if resistances_above:
            nearest_resistance = min(resistances_above, key=lambda x: x - prev["Close"])
            if (prev2["High"] >= nearest_resistance) and prev["Close"] < nearest_resistance and prev2["HighVolume"]:
                entry = curr["Open"]
                sl = entry * (1 + factor_riesgo)
                tp1 = entry * (1 - factor_riesgo)
                tp2 = entry * (1 - factor_gan_max)
                msg = f"🔴 Señal SHORT detectada en {symbol}\n" \
                      f"💰 Entrada: {entry:.2f}\n" \
                      f"🛑 Stop Loss: {sl:.2f}\n" \
                      f"🎯 TP1 (1R): {tp1:.2f}\n" \
                      f"🎯 TP2 (3R): {tp2:.2f}\n" \
                      f"📊 Parcial: {factor_venta_breack*100:.0f}% en TP1\n" \
                      f"📆 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
                send_email(msg)
                return "SHORT"

    return None


def seconds_until_next_candle(interval_minutes=240, offset_seconds=10):
    """Devuelve los segundos hasta la próxima vela (por ejemplo, 4H) menos un pequeño offset."""
    now = datetime.utcnow()
    # minutos desde el inicio del día
    minutes_since_day_start = now.hour * 60 + now.minute
    # resto de minutos desde el último múltiplo del intervalo
    remainder = minutes_since_day_start % interval_minutes
    # minutos que faltan para el próximo múltiplo
    minutes_to_next = interval_minutes - remainder
    # segundos totales hasta la próxima vela
    seconds_to_next = minutes_to_next * 60 - now.second - offset_seconds
    if seconds_to_next < 0:
        seconds_to_next += interval_minutes * 60  # por si justo pasamos el punto
    return seconds_to_next


def main():
    print("🚀 Iniciando bot sincronizado con velas 4H de Binance...")
    last_signal = None

    while True:
        try:
            signal = check_signals()
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

            if signal and signal != last_signal:
                last_signal = signal
            else:
                print(f"{now} | Sin nueva señal")

            # Calcular segundos hasta la próxima vela
            wait_seconds = seconds_until_next_candle(interval_minutes=240, offset_seconds=10)
            next_time = datetime.utcnow() + timedelta(seconds=wait_seconds)
            print(f"⏳ Esperando hasta {next_time.strftime('%Y-%m-%d %H:%M:%S UTC')} "
                  f"({wait_seconds/60:.1f} min aprox.)...\n")

            time.sleep(wait_seconds)

        except Exception as e:
            print(f"⚠️ Error en ciclo principal: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()



