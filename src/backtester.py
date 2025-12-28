import pandas as pd
import numpy as np
# On remplace matplotlib par plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.market_data import MarketData
from src.structured_products import PhoenixStructure
from src.pricing_model import EuropeanOption 

class DeltaHedgingEngine:
    def __init__(self, ticker, start_date, end_date, product_params):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.product_params = product_params
        
        # État du Portefeuille
        self.cash = 0.0           
        self.shares_held = 0.0    
        self.portfolio_value = [] 
        self.history = []         
        self.spot_series = None 
        self.attribution_history = [] 

    def fetch_data(self):
        data = MarketData.get_historical_data(self.ticker, self.start_date, self.end_date)
        if data is None or data.empty:
            raise ValueError(f"CRITICAL: No data found for {self.ticker}.")
        self.spot_series = data
        print(f"--> Backtest Engine: Loaded {len(data)} trading days.")
        return data

    def run_simulation(self):
        if self.spot_series is None:
            self.fetch_data()
            
        print(f"--- Starting Delta Hedging ({self.product_params.get('type', 'Unknown')}) ---")
        
        # 1. Initial Setup
        initial_spot = self.spot_series.iloc[0]
        params = self.product_params
        
        # Déduction du type si non explicite
        if 'type' not in params:
            prod_type = 'phoenix' if 'coupon_rate' in params else 'call'
        else:
            prod_type = params.get('type', 'phoenix').lower()
        
        # Variables spécifiques au produit
        fixed_auto_lvl, fixed_prot_lvl, fixed_coup_lvl = 0, 0, 0
        strike = 0
        nominal = initial_spot 

        # CONFIGURATION INITIALE SELON LE TYPE DE PRODUIT
        if prod_type == 'phoenix':
            fixed_auto_lvl = initial_spot * params.get('autocall_barrier', 1.0)
            fixed_prot_lvl = initial_spot * params.get('protection_barrier', 0.6)
            fixed_coup_lvl = initial_spot * params.get('coupon_barrier', 0.6)
        
        elif prod_type in ['call', 'put']:
            if 'strike_pct' in params:
                strike = initial_spot * params['strike_pct']
            else:
                strike = params.get('K', initial_spot)
            print(f"Product: {prod_type.upper()} | Strike: {strike:.2f}")

        # Initialize History Containers & Previous State
        self.attribution_history = []
        self.portfolio_value = []
        self.history = []
        
        prev_spot = initial_spot
        prev_price = 0.0
        prev_delta = 0.0
        prev_gamma = 0.0
        prev_vega = 0.0
        prev_theta = 0.0
        
        current_vol = params.get('vol', params.get('sigma', 0.20))
        
        # --- BOUCLE PRINCIPALE ---
        for i, (date, current_spot) in enumerate(self.spot_series.items()):
            
            # A. Time to Maturity
            days_remaining = len(self.spot_series) - i
            T_current = max(days_remaining / 252.0, 0.001) 
            
            # B. Calculate Greeks (ROUTAGE INTELLIGENT)
            ctx = {
                'spot': current_spot, 'T': T_current, 'vol': current_vol,
                'auto': fixed_auto_lvl, 'prot': fixed_prot_lvl, 'coup': fixed_coup_lvl,
                'nom': nominal, 'strike': strike, 'type': prod_type,
                'params': params 
            }
            
            price, delta, gamma, vega, theta = self._calculate_greeks_at_date(ctx)
            
            # C. Trading Logic
            if i == 0:
                # DAY 0
                self.cash += price 
                self.shares_held = delta
                cost_of_hedge = self.shares_held * current_spot
                self.cash -= cost_of_hedge
                
                prev_price, prev_delta, prev_gamma, prev_vega, prev_theta, prev_spot = \
                    price, delta, gamma, vega, theta, current_spot
                
            else:
                # --- P&L ATTRIBUTION ---
                dS = current_spot - prev_spot
                pnl_delta = prev_delta * dS
                pnl_gamma = - 0.5 * prev_gamma * (dS**2) 
                pnl_vega  = prev_vega * 0 
                pnl_theta = -prev_theta 
                
                predicted_pnl = pnl_delta + pnl_gamma + pnl_vega + pnl_theta
                actual_pnl_product = price - prev_price
                unexplained = actual_pnl_product - predicted_pnl
                
                self.attribution_history.append({
                    "Date": date, "Actual_PnL": actual_pnl_product, "Predicted_PnL": predicted_pnl,
                    "Delta_PnL": pnl_delta, "Gamma_PnL": pnl_gamma, "Theta_PnL": pnl_theta, "Unexplained": unexplained
                })
                
                # --- REBALANCING ---
                target_shares = delta
                trade_size = target_shares - self.shares_held
                self.cash -= (trade_size * current_spot)
                self.shares_held = target_shares
                
                # Update Prev State
                prev_price, prev_delta, prev_gamma, prev_vega, prev_theta, prev_spot = \
                    price, delta, gamma, vega, theta, current_spot

            # D. Mark-to-Market
            assets = self.cash + (self.shares_held * current_spot)
            liability = price
            pnl = assets - liability
            self.portfolio_value.append(pnl)
            
            self.history.append({
                "Date": date, "Spot": current_spot, "Model_Price": price,
                "Delta": delta, "Shares": self.shares_held, "Cash": self.cash, "PnL": pnl
            })

    def _calculate_greeks_at_date(self, ctx):
        p = ctx['params']
        s = ctx['spot']
        t = ctx['T']
        vol = ctx['vol']
        prod_type = ctx['type']

        if prod_type in ['call', 'put']:
            opt = EuropeanOption(
                S=s, K=ctx['strike'], T=t, 
                r=p.get('r', 0.05), sigma=vol, q=p.get('q', 0.0), 
                option_type=prod_type
            )
            return opt.price(), opt.delta(), opt.gamma(), opt.vega_point(), opt.daily_theta()

        elif prod_type == 'phoenix':
            def pricing_kernel(spot_val, vol_val, time_val):
                safe_s = spot_val if spot_val > 0 else 1.0
                rel_auto = ctx['auto'] / safe_s
                rel_prot = ctx['prot'] / safe_s
                rel_coup = ctx['coup'] / safe_s
                
                phx = PhoenixStructure(
                    S=spot_val, T=max(time_val, 0.001), 
                    r=p.get('r', 0.05), sigma=vol_val, q=p.get('q', 0.0),
                    coupon_rate=p.get('coupon_rate', 0.08),
                    autocall_barrier=rel_auto,
                    protection_barrier=rel_prot,
                    coupon_barrier=rel_coup, 
                    obs_frequency=p.get('obs_frequency', 4), 
                    num_simulations=2000, 
                    seed=42
                )
                return phx.price()

            # Différences Finies
            eps = s * 0.01
            p_base = pricing_kernel(s, vol, t)
            p_up   = pricing_kernel(s + eps, vol, t)
            p_down = pricing_kernel(s - eps, vol, t)
            
            delta = (p_up - p_down) / (2 * eps)
            gamma = (p_up - 2*p_base + p_down) / (eps**2)
            
            p_vol_up = pricing_kernel(s, vol + 0.01, t)
            vega = (p_vol_up - p_base) 
            
            dt = 1/252.0
            p_tomorrow = pricing_kernel(s, vol, t - dt)
            theta = p_tomorrow - p_base 

            return p_base, delta, gamma, vega, theta
            
        else:
            raise ValueError(f"Unknown product type: {prod_type}")

    # ==========================================================================
    # MODIFICATIONS : VERSIONS PLOTLY (INTERACTIVES)
    # ==========================================================================

    def plot_pnl(self):
        """
        Visualizes the Trader's Performance (Plotly Version).
        Graph 1: Spot (Left) vs Hedge (Right)
        Graph 2: Cumulative P&L
        """
        df = pd.DataFrame(self.history)
        if df.empty: return None
        
        # Création de subplots : 2 Lignes, 1 Colonne
        # La ligne 1 a deux axes Y (secondary_y=True)
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.1,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
            subplot_titles=("Market Movement & Hedge Adjustments", "Trader's Cumulative P&L")
        )

        # --- GRAPH 1 : SPOT vs HEDGE ---
        # Spot (Axe Gauche - Bleu)
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Spot'], 
            name="Spot Price", mode='lines', 
            line=dict(color='#1f77b4', width=2) # Bleu standard Matplotlib
        ), row=1, col=1, secondary_y=False)

        # Hedge/Delta (Axe Droit - Violet pointillé)
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Delta'], 
            name="Hedge (Delta)", mode='lines', 
            line=dict(color='#9467bd', width=2, dash='dash') # Violet Matplotlib
        ), row=1, col=1, secondary_y=True)

        # --- GRAPH 2 : P&L ---
        # P&L (Vert)
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['PnL'], 
            name="Cumulative P&L", mode='lines',
            line=dict(color='#2ca02c', width=2) # Vert standard Matplotlib
        ), row=2, col=1)

        # Ligne Zéro sur le P&L
        fig.add_hline(y=0, line_dash="dot", line_color="white", row=2, col=1)

        # Mise en page
        fig.update_layout(
            template="plotly_dark",
            height=700,
            hovermode="x unified",
            showlegend=True
        )
        
        # Labels Axes
        fig.update_yaxes(title_text="Spot Price", color='#1f77b4', row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Hedge (Delta)", color='#9467bd', row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Cumulative P&L (€)", row=2, col=1)

        return fig

    def plot_attribution(self):
        """
        Generates a dashboard to explain P&L sources (Plotly Version).
        Graph 1: Daily Actual vs Predicted (avec zone d'erreur)
        Graph 2: Cumulative P&L Drivers (Delta, Gamma, Theta...)
        """
        if not hasattr(self, 'attribution_history') or not self.attribution_history:
            return None

        df = pd.DataFrame(self.attribution_history)
        df['Date'] = pd.to_datetime(df['Date'])
        
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Model Accuracy: Actual vs Predicted Daily P&L", "Cumulative P&L Attribution")
        )

        # --- GRAPH 1 : DAILY ACCURACY ---
        # Predicted (Bleu pointillé)
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Predicted_PnL'],
            name="Predicted (Greeks)",
            line=dict(color='blue', dash='dash')
        ), row=1, col=1)

        # Actual (Noir)
        # Astuce Plotly pour le "Fill Between" : On remplit vers la trace précédente ('tonexty')
        # Pour simuler l'erreur rouge, on triche un peu visuellement ou on affiche les deux lignes
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Actual_PnL'],
            name="Actual Product Change",
            line=dict(color='white'), # Blanc sur fond noir ressort mieux que noir
            fill='tonexty', # Remplit l'espace entre cette ligne et la précédente (Predicted)
            fillcolor='rgba(255, 0, 0, 0.2)' # Rouge transparent pour l'erreur
        ), row=1, col=1)

        # --- GRAPH 2 : CUMULATIVE DRIVERS ---
        cols = ['Delta_PnL', 'Gamma_PnL', 'Theta_PnL', 'Unexplained']
        df_cum = df.set_index('Date')[cols].cumsum().reset_index()

        # Delta P&L (Violet)
        fig.add_trace(go.Scatter(
            x=df_cum['Date'], y=df_cum['Delta_PnL'],
            name="Delta P&L", line=dict(color='purple')
        ), row=2, col=1)

        # Gamma P&L (Orange)
        fig.add_trace(go.Scatter(
            x=df_cum['Date'], y=df_cum['Gamma_PnL'],
            name="Gamma P&L", line=dict(color='orange')
        ), row=2, col=1)

        # Theta P&L (Vert)
        fig.add_trace(go.Scatter(
            x=df_cum['Date'], y=df_cum['Theta_PnL'],
            name="Theta P&L", line=dict(color='green')
        ), row=2, col=1)

        # Unexplained (Gris pointillé)
        fig.add_trace(go.Scatter(
            x=df_cum['Date'], y=df_cum['Unexplained'],
            name="Unexplained", line=dict(color='gray', dash='dot')
        ), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=700,
            hovermode="x unified"
        )
        
        fig.update_yaxes(title_text="Daily Change (€)", row=1, col=1)
        fig.update_yaxes(title_text="Total P&L (€)", row=2, col=1)

        return fig