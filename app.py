import streamlit as st
from src.market_data import MarketData
from src.instruments import InstrumentFactory
from src.backtester import DeltaHedgingEngine
import pandas as pd
import datetime
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(page_title="Pricing & Hedging Engine", layout="wide")

st.title("🏦 Advanced Derivatives Pricing Engine")

# --- 1. SIDEBAR : MARKET DATA ---
st.sidebar.header("1. Market Data")
ticker = st.sidebar.text_input("Ticker (Yahoo)", value="GLE.PA")

if 'spot' not in st.session_state:
    st.session_state['spot'] = 100.0
if 'vol' not in st.session_state:
    st.session_state['vol'] = 0.20

if st.sidebar.button("Fetch Market Data"):
    with st.spinner("Fetching data..."):
        try:
            spot = MarketData.get_spot(ticker)
            vol = MarketData.get_volatility(ticker)
            st.session_state['spot'] = spot
            st.session_state['vol'] = vol
            st.success(f"Loaded: {ticker} (Spot: {spot:.2f}, Vol: {vol:.2%})")
        except:
            st.error("Error fetching data")

current_spot = st.session_state['spot']
current_vol = st.session_state['vol']

# --- 2. SIDEBAR : PRODUCT CONFIG ---
st.sidebar.header("2. Product Config")
option_choice = st.sidebar.selectbox("Option Type", ["European Call", "European Put", "Phoenix Autocall"])

type_mapping = {
    "European Call": "Call",
    "European Put": "Put",
    "Phoenix Autocall": "Phoenix Autocall"
}
selected_instrument_type = type_mapping[option_choice]

params = {}
params['S'] = st.sidebar.number_input("Spot Price", 0.0, 10000.0, float(current_spot), step=0.5)
params['r'] = st.sidebar.number_input("Risk Free Rate", 0.0, 0.2, 0.05, step=0.01)
params['sigma'] = st.sidebar.slider("Volatility", 0.05, 1.0, float(current_vol))
params['T'] = st.sidebar.number_input("Maturity (Years)", 0.1, 5.0, 1.0, step=0.1)

if selected_instrument_type == "Phoenix Autocall":
    st.sidebar.subheader("Phoenix Specifics")
    params['autocall_barrier'] = st.sidebar.slider("Autocall Barrier (%)", 80, 120, 100) / 100
    params['coupon_rate'] = st.sidebar.number_input("Coupon Rate", 0.0, 0.5, 0.08, step=0.01)
    params['protection_barrier'] = st.sidebar.slider("Protection Barrier (%)", 30, 90, 60) / 100
    params['coupon_barrier'] = st.sidebar.slider("Coupon Barrier (%)", 30, 90, 60) / 100
    freq_choice = st.sidebar.selectbox("Observation Freq.", ["Mensuel (12)", "Trimestriel (4)", "Semestriel (2)", "Annuel (1)"], index=1)
    params['obs_frequency'] = int(freq_choice.split("(")[1].replace(")", ""))
    params['num_simulations'] = st.sidebar.slider("MC Simulations", 1000, 40000, 2000)
else:
    params['K'] = st.sidebar.number_input("Strike Price", 0.0, 10000.0, float(current_spot), step=0.5)

# --- 3. INSTANCIATION (FACTORY) ---
try:
    product = InstrumentFactory.create_instrument(selected_instrument_type, **params)
except Exception as e:
    st.error(f"Error creating instrument: {e}")
    st.stop()

# --- 4. MAIN PAGE : TABS ---
tab_structuring, tab_trading, tab_backtest = st.tabs(["📐 Structuring", "📊 Trading & Risk", "🕰️ Backtest"])

# --- ONGLET STRUCTURING ---
with tab_structuring:
    st.header("📐 Structuring & Pricing Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Fair Value")
        with st.spinner('Pricing...'):
            price = product.price()
            
            if selected_instrument_type == "Phoenix Autocall":
                price_pct = (price / params['S']) * 100
                st.metric(label="Price (% Nominal)", value=f"{price_pct:.2f} %")
            else:
                st.metric(label="Premium (€)", value=f"{price:.2f} €")

        st.info("ℹ️ Modifiez les paramètres à gauche pour rafraîchir les graphiques.")

    with col2:
        # 1. GRAPH PRINCIPAL : PAYOFF PROFILE (Toujours affiché)
        st.subheader("Payoff Profile at Maturity")
        spot_val = params['S']
        fig_payoff = product.plot_payoff([spot_val * 0.4, spot_val * 1.3])
        st.plotly_chart(fig_payoff, use_container_width=True)

        # 2. SECTION AVANCÉE (Uniquement pour Phoenix, affichée directement en dessous)
        if selected_instrument_type == "Phoenix Autocall":
            st.divider()
            st.subheader("🔬 Advanced Monte Carlo Analysis")
            
            # Onglets pour organiser les 3 graphiques supplémentaires sans surcharger la page
            subtab_tunnel, subtab_dist, subtab_noise = st.tabs(["🌪️ Scenarios (Tunnel)", "📊 Payoff Dist.", "📉 Convergence"])
            
            with subtab_tunnel:
                st.caption("Visualisation de 200 chemins aléatoires classés par résultat final.")
                with st.spinner("Generating Paths..."):
                    fig_tunnel = product.plot_phoenix_tunnel()
                    st.plotly_chart(fig_tunnel, use_container_width=True)
            
            with subtab_dist:
                st.caption("Distribution des résultats financiers pour le client (Histogramme).")
                with st.spinner("Computing Distribution..."):
                    fig_dist = product.plot_phoenix_distribution()
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
            with subtab_noise:
                st.caption("Analyse de stabilité du prix (Bruit Monte Carlo sur 30 runs).")
                # On garde le bouton pour ce calcul spécifique car il est très lourd (30 x Pricing)
                if st.button("Lancer l'analyse de convergence"):
                    with st.spinner("Running multiple pricings..."):
                        fig_noise = product.plot_mc_noise_distribution()
                        st.plotly_chart(fig_noise, use_container_width=True)
                else:
                    st.write("Le test de convergence lance 30 pricings successifs. Cliquez pour démarrer.")

    st.divider()

    # Section Price vs Strike (Uniquement pour Call/Put)
    if selected_instrument_type in ["Call", "Put"]:
        st.subheader("Structuring View: Price vs Strike")
        fig_struct = product.plot_price_vs_strike(params['S'])
        st.plotly_chart(fig_struct, use_container_width=True)

# --- ONGLET TRADING ---
with tab_trading:
    st.header("Trading & Risk Management")
    
    st.subheader("1. Instantaneous Greeks")
    with st.spinner("Calculating Risk Sensitivities..."):
        greeks = product.greeks()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Delta", f"{greeks.get('delta', 0):.2f}")
    c2.metric("Gamma", f"{greeks.get('gamma', 0):.4f}")
    c3.metric("Vega", f"{greeks.get('vega', 0):.2f}")
    c4.metric("Theta", f"{greeks.get('theta', 0):.3f}")
    
    st.divider()

    st.subheader("2. Risk Matrices (Scenario Analysis)")
    hm_col1, hm_col2, hm_col3, hm_col4 = st.columns(4)
    with hm_col1: hm_spot_range = st.slider("Spot Range (+/- %)", 5, 50, 10, step=5) / 100
    with hm_col2: hm_vol_range = st.slider("Vol Range (+/- %)", 1, 20, 5, step=1) / 100
    with hm_col3: n_spot = st.number_input("Spot Steps (X)", min_value=3, max_value=21, value=5, step=2)
    with hm_col4: n_vol = st.number_input("Vol Steps (Y)", min_value=3, max_value=21, value=3, step=2)

    if st.checkbox("Show Heatmaps", value=True):
        with st.spinner("Computing Dynamic Risk Matrices..."):
            fig_matrix = product.plot_risk_matrix(
                spot_range_pct=hm_spot_range, 
                vol_range_pct=hm_vol_range, 
                n_spot_steps=n_spot,
                n_vol_steps=n_vol
            )
            st.plotly_chart(fig_matrix, use_container_width=True)

    st.divider()

    st.subheader("3. P&L Attribution Simulator")
    col_sim_1, col_sim_2, col_sim_3 = st.columns(3)
    with col_sim_1: sim_spot_move = st.number_input("Scenario: Spot Move (%)", -50.0, 50.0, 5.0, step=0.5) / 100
    with col_sim_2: sim_vol_move = st.number_input("Scenario: Vol Move (pts)", -20.0, 20.0, 1.0, step=0.5) / 100
    with col_sim_3: sim_days = st.number_input("Days Passed (Theta)", 0, 365, 0)

    with st.spinner("Decomposing P&L..."):
        fig_pnl = product.plot_pnl_attribution(sim_spot_move, sim_vol_move, sim_days)
        st.plotly_chart(fig_pnl, use_container_width=True)

    st.divider()

    if selected_instrument_type in ["Call", "Put"]:
        st.subheader("4. Hedging Difficulties")
        current_s = params['S']
        fig_risk = product.plot_risk_profile([current_s * 0.8, current_s * 1.2])
        st.plotly_chart(fig_risk, use_container_width=True)

with tab_backtest:
    st.header("🕰️ Backtest Historique (Moteur Statique)")
    
    col_bt1, col_bt2, col_bt3 = st.columns(3)
    
    with col_bt1:
        default_start = datetime.date(2023, 6, 1)
        default_end = datetime.date(2024, 6, 1)
        start_date_input = st.date_input("Start Date", value=default_start)
    
    with col_bt2:
        end_date_input = st.date_input("End Date", value=default_end)
        
    with col_bt3:
        st.write("") 
        run_bt = st.button("Lancer le Backtest", type="primary")

    if run_bt:
        # Préparation des paramètres
        bt_params = params.copy()
        
        if selected_instrument_type == "Phoenix Autocall":
            bt_params['type'] = 'phoenix'
        else:
            bt_params['type'] = type_mapping[option_choice].lower() 
            bt_params['strike_pct'] = params['K'] / current_spot
            
        st.info(f"Simulation en cours sur {ticker} du {start_date_input} au {end_date_input}...")
        
        try:
            # 1. Instanciation
            engine = DeltaHedgingEngine(
                ticker=ticker,
                start_date=str(start_date_input),
                end_date=str(end_date_input),
                product_params=bt_params
            )
            
            # 2. Exécution
            with st.spinner("Calcul des Grecs et rebalancement quotidien..."):
                engine.run_simulation()
            
            st.success("Backtest terminé !")
            
            # 3. Affichage des graphiques (VERSION PLOTLY INTERACTIVE)
            
            st.subheader("1. Performance du Trader & Hedge")
            # Appel de la nouvelle méthode plot_pnl (qui retourne une figure Plotly)
            fig_pnl = engine.plot_pnl()
            if fig_pnl:
                st.plotly_chart(fig_pnl, use_container_width=True)
            
            st.subheader("2. Attribution de P&L (Taylor Expansion)")
            # Appel de la nouvelle méthode plot_attribution
            fig_attrib = engine.plot_attribution()
            if fig_attrib:
                st.plotly_chart(fig_attrib, use_container_width=True)
                
            # Métrique final
            if engine.history:
                final_pnl = engine.history[-1]['PnL']
                # Affichage coloré selon le signe
                st.metric("P&L Final", f"{final_pnl:.2f} €", delta=final_pnl)
                
        except Exception as e:
            st.error(f"Erreur durant le backtest : {e}")
            # Pour debugger si besoin :
            # import traceback
            # st.code(traceback.format_exc())