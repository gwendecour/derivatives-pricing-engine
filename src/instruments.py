from abc import ABC, abstractmethod
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

class FinancialInstrument(ABC):
    """
    Classe mère abstraite. 
    Force toutes les options (Call, Phoenix...) à avoir les mêmes méthodes.
    """
    def __init__(self, **params):
        # On stocke tous les paramètres (S, K, T, etc.) dans un dictionnaire
        self.params = params
        
    @abstractmethod
    def price(self) -> float:
        pass

    @abstractmethod
    def greeks(self) -> dict:
        """Doit retourner un dictionnaire ex: {'delta': 0.5, 'gamma': 0.02, ...}"""
        pass

    @abstractmethod
    def plot_payoff(self, spot_range) -> go.Figure:
        pass

    # --- MISE À JOUR : MATRICES DE RISQUE DYNAMIQUES ---
    def plot_risk_matrix(self, spot_range_pct=0.10, vol_range_pct=0.05, n_spot_steps=5, n_vol_steps=3):
        """
        Génère les Heatmaps avec dimensions X/Y indépendantes.
        """
        # 1. Génération dynamique des axes (Indépendants)
        spot_moves = np.linspace(-spot_range_pct, spot_range_pct, n_spot_steps)
        vol_moves = np.linspace(-vol_range_pct, vol_range_pct, n_vol_steps)
        
        # Inversion de l'axe Volatilité pour avoir le + haut en haut (standard graphique)
        # Mais attention, heatmap de Plotly met souvent l'indice 0 en bas.
        # On garde linspace tel quel, Plotly gère l'axe Y correctement avec les labels.
        
        # Sauvegarde état initial
        original_S = self.S
        original_sigma = self.sigma
        
        base_price = self.price()
        base_greeks = self.greeks() 
        base_delta = base_greeks.get('delta', 0.0)
        
        # Matrices rectangulaires (Rows=Vol, Cols=Spot)
        z_unhedged = np.zeros((len(vol_moves), len(spot_moves)))
        z_hedged = np.zeros((len(vol_moves), len(spot_moves)))
        
        # 2. Boucle de calcul
        for i, v_chg in enumerate(vol_moves):
            for j, s_chg in enumerate(spot_moves):
                self.S = original_S * (1 + s_chg)
                self.sigma = max(0.01, original_sigma + v_chg)
                
                new_price = self.price()
                pnl_option = -(new_price - base_price) 
                pnl_hedge = base_delta * (self.S - original_S)
                
                z_unhedged[i, j] = pnl_option
                z_hedged[i, j] = pnl_option + pnl_hedge

        # Reset
        self.S = original_S
        self.sigma = original_sigma
        
        # 3. Graphiques
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=("1. P&L Non-Couvert (Directionnel)", "2. P&L Delta-Hedgé (Gamma/Vega)"),
            horizontal_spacing=0.15
        )

        x_labels = [f"{m*100:+.1f}%" for m in spot_moves]
        y_labels = [f"{v*100:+.1f}%" for v in vol_moves]

        # Heatmap 1 : Unhedged
        fig.add_trace(go.Heatmap(
            z=z_unhedged, x=x_labels, y=y_labels,
            colorscale='RdYlGn', zmid=0, 
            showscale=True, 
            colorbar=dict(title="P&L (€)", x=-0.15),
            # CORRECTION ICI : .2f pour avoir les décimales
            texttemplate="%{z:.2f}", textfont={"size":10} 
        ), row=1, col=1)

        # Heatmap 2 : Hedged
        fig.add_trace(go.Heatmap(
            z=z_hedged, x=x_labels, y=y_labels,
            colorscale='RdYlGn', zmid=0, 
            showscale=True,
            texttemplate="%{z:.2f}", textfont={"size":10},
            colorbar=dict(title="P&L (€)", x=1.02)
        ), row=1, col=2)

        fig.update_layout(
            title="Matrices de Risque Dynamiques",
            xaxis_title="Variation Spot", 
            yaxis_title="Variation Volatilité",
            template="plotly_dark",
            height=500
        )
        fig.update_xaxes(title_text="Variation Spot", row=1, col=2)
        
        return fig
    
    # --- MISE À JOUR : P&L ATTRIBUTION (BAR CHART) ---
    def plot_pnl_attribution(self, spot_move_pct, vol_move_pct, days_passed=0):
        """
        Explique le P&L via Taylor Expansion (Delta, Gamma, Vega, Theta).
        Affichage : Bar Chart simple (Vert/Rouge/Bleu).
        """
        # 1. Calculs (inchangés)
        original_S, original_sigma, original_T = self.S, self.sigma, self.T
        base_price = self.price()
        greeks = self.greeks()
        
        dt = days_passed / 365.0
        dS = original_S * spot_move_pct
        # Pour Vega : on suppose que greeks['vega'] est pour 1% (0.01) de vol
        # Si vol_move_pct = 0.01, impact = 1 * Vega.
        # Donc impact = Vega * (vol_move_pct * 100)
        pnl_vega = (greeks['vega'] * (vol_move_pct * 100)) * -1 
        
        pos_sign = -1 # Short Option
        pnl_delta = (greeks['delta'] * dS) * pos_sign
        pnl_gamma = (0.5 * greeks['gamma'] * (dS**2)) * pos_sign
        pnl_theta = (greeks['theta'] * days_passed) * pos_sign
        
        predicted_pnl = pnl_delta + pnl_gamma + pnl_vega + pnl_theta
        
        # Repricing réel
        self.S = original_S * (1 + spot_move_pct)
        self.sigma = original_sigma + vol_move_pct
        self.T = max(0.001, original_T - dt)
        
        new_price = self.price()
        actual_pnl = (new_price - base_price) * pos_sign
        unexplained = actual_pnl - predicted_pnl
        
        # Reset
        self.S, self.sigma, self.T = original_S, original_sigma, original_T
        
        # 2. Préparation des données pour le Graphique
        categories = ["Delta", "Gamma", "Vega", "Theta", "Unexplained", "Predicted Total", "Actual Total"]
        values = [pnl_delta, pnl_gamma, pnl_vega, pnl_theta, unexplained, predicted_pnl, actual_pnl]
        
        # 3. Gestion des Couleurs
        # Vert si > 0, Rouge si < 0.
        # Pour les totaux (les deux derniers), on force Bleu.
        colors = []
        for i, val in enumerate(values):
            if i >= 5: # Les deux dernières barres (Totaux)
                colors.append('#3366CC') # Bleu
            else:
                colors.append('#2ECC40' if val >= 0 else '#FF4136') # Vert ou Rouge
        
        # 4. Création du Bar Chart
        fig = go.Figure(go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:.2f} €" for v in values],
            textposition='auto'
        ))

        fig.add_hline(y=0, line_color="white", line_width=1)

        fig.update_layout(
            title=f"P&L Attribution (Spot {spot_move_pct:+.1%}, Vol {vol_move_pct:+.1%}, {days_passed}j)",
            template="plotly_dark",
            yaxis_title="Profit / Loss (€)",
            showlegend=False
        )

        return fig

class InstrumentFactory:
    """
    Le chef d'orchestre qui crée le bon objet selon le choix de l'utilisateur.
    """
    @staticmethod
    def create_instrument(instrument_type, **kwargs):
        # Importations locales pour éviter les erreurs d'import circulaire
        from src.pricing_model import EuropeanOption
        from src.structured_products import PhoenixStructure

        if instrument_type in ["Call", "Put"]:
            # On passe 'option_type' (call/put) en plus des autres params
            return EuropeanOption(option_type=instrument_type.lower(), **kwargs)
        
        elif instrument_type == "Phoenix Autocall":
            return PhoenixStructure(**kwargs)
        
        else:
            raise ValueError(f"Instrument inconnu : {instrument_type}")