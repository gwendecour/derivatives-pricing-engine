import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from src.instruments import FinancialInstrument

class EuropeanOption(FinancialInstrument):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.S = float(kwargs.get('S'))
        self.K = float(kwargs.get('K'))
        self.T = float(kwargs.get('T'))
        self.r = float(kwargs.get('r'))
        self.sigma = float(kwargs.get('sigma'))
        self.q = float(kwargs.get('q', 0.0)) 
        self.option_type = kwargs.get('option_type', 'call').lower()

    def _d1(self):
        return (np.log(self.S/self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    def _d2(self):
        return self._d1() - self.sigma * np.sqrt(self.T)

    def price(self):
        d1 = self._d1()
        d2 = self._d2()
        if self.option_type == "call":
            return self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)
        
    def greeks(self):
        return {
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega": self.vega_point(),
            "theta": self.daily_theta(),
            "rho": self.rho_point()
        }

    def delta(self):
        if self.option_type == "call":
            return np.exp(-self.q * self.T) * norm.cdf(self._d1())
        else:
            return -np.exp(-self.q * self.T) * norm.cdf(-self._d1())

    def gamma(self):
        return np.exp(-self.q * self.T) * norm.pdf(self._d1()) / (self.S * self.sigma * np.sqrt(self.T))

    def vega_point(self):
        # Value for a 1% change in Volatility
        return (self.S * np.exp(-self.q * self.T) * norm.pdf(self._d1()) * np.sqrt(self.T)) / 100

    def daily_theta(self):
        # Value for 1 Day time decay
        d1 = self._d1()
        d2 = self._d2()
        common = -(self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        
        if self.option_type == "call":
            theta = common - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2) + self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(d1)
        else:
            theta = common + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)
            
        return theta / 365

    def rho_point(self):
        # Value for 1% change in Rates
        d2 = self._d2()
        if self.option_type == "call":
            return (self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)) / 100
        else:
            return -(self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)) / 100
    
    def plot_payoff(self, spot_range):
        """
        Génère un graphique interactif Plotly montrant le P&L du Client vs Banque.
        """
        spots = np.linspace(spot_range[0], spot_range[1], 100)
        premium = self.price()
        
        if self.option_type == "call":
            intrinsic_value = np.maximum(spots - self.K, 0)
        else:
            intrinsic_value = np.maximum(self.K - spots, 0)

        pnl_client = intrinsic_value - premium
        pnl_bank = premium - intrinsic_value
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=spots, y=pnl_client, 
            mode='lines', 
            name=f'Client (Long {self.option_type.title()})', 
            line=dict(color='green', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=spots, y=pnl_bank, 
            mode='lines', 
            name=f'Bank (Short {self.option_type.title()})', 
            line=dict(color='red', width=3)
        ))

        fig.add_hline(y=0, line_color="white", line_width=1, opacity=0.5)

        fig.add_vline(
            x=self.K, 
            line_dash="dash", line_color="gray", 
            annotation_text=f"Strike ({self.K:.1f})", annotation_position="top left"
        )

        fig.add_vline(
            x=self.S, 
            line_dash="dot", line_color="cyan", 
            annotation_text=f"Spot Actuel ({self.S:.1f})", annotation_position="bottom right"
        )

        fig.update_layout(
            title=f"Profil de P&L à Maturité (Prime = {premium:.2f}€)",
            xaxis_title="Prix du Sous-jacent à Maturité",
            yaxis_title="Profit / Perte (€)",
            template="plotly_dark", # Thème sombre pour faire ressortir le Vert/Rouge
            hovermode="x unified",   # Pour voir les deux valeurs en même temps au survol
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def plot_price_vs_strike(self, current_spot):
        """
        Affiche le prix de l'option en fonction du Strike (K).
        Ajoute un point rouge interactif indiquant la position actuelle.
        """
        # 1. Plage de Strikes (de 50% à 150% du Spot actuel)
        strikes = np.linspace(current_spot * 0.5, current_spot * 1.5, 100)
        prices = []
        
        # 2. Calcul du prix pour chaque Strike simulé
        # On garde les mêmes paramètres (vol, r, T...) sauf K qui change
        for k in strikes:
            temp_opt = EuropeanOption(
                S=self.S, K=k, T=self.T, r=self.r, sigma=self.sigma, q=self.q, option_type=self.option_type
            )
            prices.append(temp_opt.price())
            
        # 3. Récupération du point actuel (Notre Strike choisi)
        current_price = self.price()
        current_k = self.K
        
        # 4. Construction du Graphique
        fig = go.Figure()
        
        # La courbe bleue (Tous les prix possibles)
        fig.add_trace(go.Scatter(
            x=strikes, y=prices, 
            mode='lines', 
            name='Prix Théorique',
            line=dict(color='royalblue', width=2)
        ))
        
        # Le point rouge (Notre configuration actuelle)
        fig.add_trace(go.Scatter(
            x=[current_k], y=[current_price],
            mode='markers',
            name='Votre Sélection',
            marker=dict(color='red', size=12, line=dict(color='white', width=2))
        ))
        
        # Lignes guides
        fig.add_vline(x=current_spot, line_dash="dot", line_color="gray", annotation_text="Spot Actuel")

        fig.update_layout(
            title="Sensibilité du Prix au Strike (Moneyness)",
            xaxis_title="Strike Price (K)",
            yaxis_title="Prime de l'Option (€)",
            template="plotly_dark",
            hovermode="x unified",
            showlegend=True
        )
        
        return fig
    
    def plot_risk_profile(self, spot_range):
        """
        Affiche les risques de couverture (Gamma & Vega) en fonction du Spot.
        C'est la vue "Hedging Difficulty".
        """
        # 1. Génération des scénarios de marché (Spot +/- 20%)
        spots = np.linspace(spot_range[0], spot_range[1], 100)
        gammas = []
        vegas = []
        
        # 2. Calcul des Grecs pour chaque scénario
        for s in spots:
            # On simule : Si le spot valait 's', quels seraient mes risques ?
            temp_opt = EuropeanOption(
                S=s, K=self.K, T=self.T, r=self.r, sigma=self.sigma, q=self.q, option_type=self.option_type
            )
            gammas.append(temp_opt.gamma())
            vegas.append(temp_opt.vega_point()) # Vega pour 1% de vol

        # 3. Récupération des valeurs actuelles pour le point rouge
        current_gamma = self.gamma()
        current_vega = self.vega_point()

        # 4. Construction du Graphique Double Axe
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # --- TRACE 1 : GAMMA (Axe Gauche - Rouge) ---
        fig.add_trace(
            go.Scatter(x=spots, y=gammas, mode='lines', name='Gamma (Convexity)', line=dict(color='crimson', width=3)),
            secondary_y=False
        )
        
        # --- TRACE 2 : VEGA (Axe Droit - Bleu pointillé) ---
        fig.add_trace(
            go.Scatter(x=spots, y=vegas, mode='lines', name='Vega (Vol Risk)', line=dict(color='royalblue', width=2, dash='dash')),
            secondary_y=True
        )

        # --- POINTS ACTUELS (Pour se situer) ---
        fig.add_trace(
            go.Scatter(x=[self.S], y=[current_gamma], mode='markers', name='Mon Gamma', marker=dict(color='crimson', size=10)),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=[self.S], y=[current_vega], mode='markers', name='Mon Vega', marker=dict(color='royalblue', size=10)),
            secondary_y=True
        )

        # 5. Mise en page
        fig.update_layout(
            title="Hedging Difficulties: Gamma & Vega Sensitivity",
            xaxis_title="Spot Price (Scenarios)",
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Configuration des Axes Y
        fig.update_yaxes(title_text="Gamma", title_font=dict(color="crimson"), tickfont=dict(color="crimson"), secondary_y=False)
        fig.update_yaxes(title_text="Vega", title_font=dict(color="royalblue"), tickfont=dict(color="royalblue"), secondary_y=True)
        
        # Ligne verticale du Spot actuel
        fig.add_vline(x=self.S, line_dash="dot", line_color="gray", annotation_text="Spot Actuel")

        return fig