import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Configuración de la página de Streamlit
st.set_page_config(page_title="Fintech Analytics Platform", layout="wide", page_icon="📈")
st.title("📈 Plataforma Inteligente de Portafolio y Screener")

# Crear pestañas principales
tab1, tab2 = st.tabs(["⚖️ Optimizador de Portafolio", "🔍 Screener de Acciones"])

# ==========================================
# PESTAÑA 1: OPTIMIZADOR & FRONTERA EFICIENTE
# ==========================================
with tab1:
    st.header("Optimización de Portafolio (Teoría de Markowitz)")
    
    # Inputs del usuario en la barra lateral de esta pestaña
    col_input1, col_input2 = st.columns([1, 2])
    with col_input1:
        tickers_input = st.text_input("Ingresa los Tickers separados por coma:", "AAPL, MSFT, GOOGL, AMZN, JPM")
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        
        num_simulaciones = st.slider("Número de simulaciones de Monte Carlo", 1000, 20000, 5000, step=1000)
        tasa_libre_riesgo = st.number_input("Tasa libre de riesgo (ej: 0.042 para 4.2%)", value=0.042, format="%.3f")

    if st.button("🚀 Ejecutar Optimización"):
        with st.spinner("Descargando datos y simulando portafolios..."):
            try:
                # Descarga de datos
                datos = yf.download(tickers, period="3y")["Adj Close"]
                retornos_diarios = datos.pct_change().dropna()
                retornos_anuales = retornos_diarios.mean() * 252
                matriz_covarianza = retornos_diarios.cov() * 252

                # Simulación de Monte Carlo
                resultados = np.zeros((3 + len(tickers), num_simulaciones))
                for i in range(num_simulaciones):
                    pesos = np.array(np.random.random(len(tickers)))
                    pesos /= np.sum(pesos)

                    retorno_p = np.sum(retornos_anuales * pesos)
                    volatilidad_p = np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos)))
                    sharpe_p = (retorno_p - tasa_libre_riesgo) / volatilidad_p

                    resultados[0, i] = retorno_p
                    resultados[1, i] = volatilidad_p
                    resultados[2, i] = sharpe_p
                    for j in range(len(pesos)):
                        resultados[3 + j, i] = pesos[j]

                # Dataframe con resultados de las simulaciones
                df_sim = pd.DataFrame(resultados.T, columns=["Retorno", "Volatilidad", "Sharpe"] + tickers)
                
                # Identificar Portafolio Óptimo (Max Sharpe) y Mínima Volatilidad
                idx_max_sharpe = df_sim["Sharpe"].idxmax()
                portafolio_optimo = df_sim.iloc[idx_max_sharpe]
                
                idx_min_vol = df_sim["Volatilidad"].idxmin()
                portafolio_min_vol = df_sim.iloc[idx_min_vol]

                # --- Métricas Clave ---
                st.subheader("💡 Resultados del Análisis")
                m1, m2, m3 = st.columns(3)
                m1.metric("Retorno Esperado (Max Sharpe)", f"{portafolio_optimo['Retorno']*100:.2f}%")
                m2.metric("Volatilidad / Riesgo", f"{portafolio_optimo['Volatilidad']*100:.2f}%")
                m3.metric("Ratio de Sharpe Máximo", f"{portafolio_optimo['Sharpe']:.2f}")

                # --- Gráfico de Frontera Eficiente ---
                st.subheader("📊 Visualización de la Frontera Eficiente")
                
                fig = px.scatter(
                    df_sim, x="Volatilidad", y="Retorno", color="Sharpe",
                    labels={"Volatilidad": "Riesgo (Volatilidad Anual)", "Retorno": "Retorno Anual Esperado"},
                    title="Simulaciones de Portafolio", color_continuous_scale="Viridis", opacity=0.5
                )
                
                # Añadir punto del portafolio óptimo
                fig.add_trace(go.Scatter(
                    x=[portafolio_optimo["Volatilidad"]], y=[portafolio_optimo["Retorno"]],
                    mode="markers", name="Max Sharpe",
                    marker=dict(color="red", size=15, symbol="star")
                ))
                
                # Añadir punto de mínima volatilidad
                fig.add_trace(go.Scatter(
                    x=[portafolio_min_vol["Volatilidad"]], y=[portafolio_min_vol["Retorno"]],
                    mode="markers", name="Mínima Volatilidad",
                    marker=dict(color="orange", size=12, symbol="diamond")
                ))

                st.plotly_chart(fig, use_container_width=True)

                # --- Pesos Sugeridos ---
                st.subheader("⚖️ Pesos Sugeridos para tu Cartera")
                pesos_df = pd.DataFrame({
                    "Activo": tickers,
                    "Peso Optimizando Sharpe (Max Rendimiento/Riesgo)": [f"{portafolio_optimo[t]*100:.1f}%" for t in tickers],
                    "Peso Mínima Volatilidad (Conservador)": [f"{portafolio_min_vol[t]*100:.1f}%" for t in tickers]
                })
                st.dataframe(pesos_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error al procesar los tickers: {e}. Asegúrate de escribir tickers válidos de Yahoo Finance.")

# ==========================================
# PESTAÑA 2: SCREENER DE ACCIONES
# ==========================================
with tab2:
    st.header("🔍 Buscador y Filtro de Acciones (Screener)")
    st.write("Filtra rápidamente una lista predefinida de empresas según sus fundamentales.")
    
    # Lista por defecto para el screener (puedes ampliarla)
    screener_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V", "DIS", "NFLX", "AMD"]
    
    # Controles de filtrado interactivo
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        max_pe = st.slider("Ratio P/E Máximo (Trailing)", 10, 100, 40, step=5)
    with col_f2:
        min_div = st.slider("Rendimiento por Dividendo Mínimo (%)", 0.0, 5.0, 0.0, step=0.5)
    with col_f3:
        ordenar_por = st.selectbox("Ordenar resultados por:", ["Trailing P/E", "Dividend Yield (%)", "Margen Operativo (%)"])

    if st.button("🔍 Ejecutar Escaneo"):
        with st.spinner("Escaneando fundamentales del mercado..."):
            datos_screener = []
            for ticker in screener_tickers:
                try:
                    t = yf.Ticker(ticker)
                    info = t.info
                    
                    pe = info.get("trailingPE", np.nan)
                    div = info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0.0
                    margin = info.get("operatingMargins", 0) * 100 if info.get("operatingMargins") else 0.0
                    price = info.get("currentPrice", np.nan)
                    
                    datos_screener.append({
                        "Ticker": ticker,
                        "Nombre": info.get("longName", "N/A"),
                        "Precio": price,
                        "Trailing P/E": pe,
                        "Dividend Yield (%)": div,
                        "Margen Operativo (%)": margin
                    })
                except:
                    continue
            
            df_screener = pd.DataFrame(datos_screener)
            
            # Aplicar Filtros de usuario
            df_filtrado = df_screener[
                (df_screener["Trailing P/E"] <= max_pe) | (df_screener["Trailing P/E"].isna())
            ]
            df_filtrado = df_filtrado[df_filtrado["Dividend Yield (%)"] >= min_div]
            
            # Ordenar
            ascendente = True if ordenar_por == "Trailing P/E" else False
            df_filtrado = df_filtrado.sort_values(by=ordenar_por, ascending=ascendente)
            
            st.subheader(f"📋 Acciones que cumplen tus criterios ({len(df_filtrado)} encontradas)")
            st.dataframe(df_filtrado.style.format({
                "Precio": "${:.2f}",
                "Trailing P/E": "{:.2f}",
                "Dividend Yield (%)": "{:.2f}%",
                "Margen Operativo (%)": "{:.2f}%"
            }), use_container_width=True)
