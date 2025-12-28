import numpy as np
import plotly.graph_objects as go
from src.monte_carlo import MonteCarloEngine
from src.instruments import FinancialInstrument
import plotly.express as px

class PhoenixStructure(MonteCarloEngine, FinancialInstrument):
    
    def __init__(self, **kwargs):
        # 1. Récupération des paramètres
        S = float(kwargs.get('S'))
        self.nominal = S
        self.coupon_rate = kwargs.get('coupon_rate')
        
        # Gestion des barrières (entrée en % ou absolu, ici on standardise en absolu)
        self.autocall_barrier = S * kwargs.get('autocall_barrier')
        self.protection_barrier = S * kwargs.get('protection_barrier')
        self.coupon_barrier = S * kwargs.get('coupon_barrier')
        
        # Fréquence d'observation (Défaut à 4 = Trimestriel)
        self.obs_frequency = kwargs.get('obs_frequency', 4)
        
        maturity = float(kwargs.get('T'))
        
        # 2. Calcul du nombre de steps pour le Monte Carlo
        steps = max(int(252 * maturity), 1)
        self.steps = steps
        
        num_simulations = kwargs.get('num_simulations', 10000)
        
        # 3. Init Moteur Monte Carlo (Parent 1)
        MonteCarloEngine.__init__(self, 
            S=S, K=S, T=maturity, 
            r=kwargs.get('r'), 
            sigma=kwargs.get('sigma'), 
            q=kwargs.get('q', 0.0), 
            num_simulations=num_simulations, 
            num_steps=steps, 
            seed=kwargs.get('seed')
        )
        
        # 4. Init Instrument (Parent 2)
        FinancialInstrument.__init__(self, **kwargs)

    def get_observation_indices(self):
        step_size = int(252 / self.obs_frequency)
        # On s'assure de ne pas dépasser le nombre de steps total
        indices = np.arange(step_size, self.steps + 1, step_size, dtype=int)
        return indices

    def calculate_payoffs_distribution(self):
        # Ta logique existante (Back-end)
        paths = self.generate_paths() 
        payoffs = np.zeros(self.N)
        active_paths = np.ones(self.N, dtype=bool)
        indices = self.get_observation_indices()
        
        coupon_amt = self.nominal * self.coupon_rate * (1.0/self.obs_frequency)
        
        for i, idx in enumerate(indices):
            if idx >= len(paths): break
            current_prices = paths[idx]
            
            # Conditions
            did_autocall = (current_prices >= self.autocall_barrier) & active_paths
            did_just_coupon = (current_prices >= self.coupon_barrier) & (current_prices < self.autocall_barrier) & active_paths
            
            # Discounting
            time_fraction = idx / 252.0
            df = np.exp(-self.r * time_fraction)
            
            # Payoff Logic
            payoffs[did_just_coupon] += coupon_amt * df
            payoffs[did_autocall] += (self.nominal + coupon_amt) * df
            
            active_paths[did_autocall] = False
            if not np.any(active_paths): break
    
        if np.any(active_paths):
            final_prices = paths[-1]
            survivors = active_paths
            df_final = np.exp(-self.r * self.T)
            
            safe_mask = survivors & (final_prices >= self.protection_barrier)
            payoffs[safe_mask] += self.nominal * df_final
            
            crash_mask = survivors & (final_prices < self.protection_barrier)
            payoffs[crash_mask] += final_prices[crash_mask] * df_final

        return payoffs

    def price(self):
        payoffs = self.calculate_payoffs_distribution()
        return np.mean(payoffs)

    def plot_payoff(self, spot_range):
        """
        Génère le graphique de profitabilité à maturité (Vision Structurer/Client).
        Montre la barrière et le saut de rendement (Coupon).
        """
        spots = np.linspace(spot_range[0], spot_range[1], 300)
        payoffs = []
        
        # Calcul du remboursement total en cas de succès (Nominal + Coupon Annuel)
        # C'est ce que le client voit : "Si c'est au dessus, je prends 110%"
        success_payoff = self.nominal * (1 + self.coupon_rate)
        
        for s in spots:
            if s < self.protection_barrier:
                # SCENARIO CRASH : On récupère l'action (perte en capital)
                payoffs.append(s)
            
            elif s < self.coupon_barrier:
                # SCENARIO NEUTRE (Rare) : Au dessus protection mais sous coupon
                # On récupère 100% du capital mais pas de coupon
                payoffs.append(self.nominal)
                
            else:
                # SCENARIO GAGNANT : Capital + Coupon
                # C'est ici qu'on crée la marche d'escalier vers 110% (ou autre)
                payoffs.append(success_payoff)
        
        # Conversion en % du nominal pour l'affichage
        payoffs_pct = (np.array(payoffs) / self.nominal) * 100
        current_price_pct = (self.price() / self.nominal) * 100
        
        fig = go.Figure()
        
        # Trace du Payoff (Cyan)
        fig.add_trace(go.Scatter(
            x=spots, y=payoffs_pct, 
            mode='lines', name='Remboursement Final', 
            line=dict(color='cyan', width=3)
        ))
        
        # Barrières
        fig.add_vline(x=self.protection_barrier, line_dash="dash", line_color="red", annotation_text="Protection")
        
        if self.coupon_barrier != self.protection_barrier:
            fig.add_vline(x=self.coupon_barrier, line_dash="dot", line_color="yellow", annotation_text="Coupon Trigger")
            
        fig.add_vline(x=self.autocall_barrier, line_dash="dash", line_color="green", annotation_text="Autocall")
        
        # Ligne 100%
        fig.add_hline(y=100, line_color="gray", line_width=1, annotation_text="Capital Initial")

        fig.update_layout(
            title=f"Profil à Maturité (Prix Actuel: {current_price_pct:.2f}%)",
            xaxis_title="Prix du Sous-jacent",
            yaxis_title="Remboursement (% Nominal)",
            template="plotly_dark",
            yaxis=dict(range=[0, max(payoffs_pct)*1.15]) # Marge en haut pour voir le coupon
        )
        return fig

    def greeks(self):
        # Utilisation de la méthode des différences finies (Bump & Revalue)
        original_seed = self.seed if self.seed else 42
        self.seed = original_seed
        base_price = self.price()
        
        # Delta & Gamma
        epsilon = self.S * 0.01 
        orig_S = self.S
        
        self.S = orig_S + epsilon
        self.seed = original_seed
        p_up = self.price()
        
        self.S = orig_S - epsilon
        self.seed = original_seed
        p_down = self.price()
        
        self.S = orig_S # Reset
        
        delta = (p_up - p_down) / (2 * epsilon)
        gamma = (p_up - 2 * base_price + p_down) / (epsilon**2)
        
        # Vega
        orig_sigma = self.sigma
        self.sigma = orig_sigma + 0.01
        self.seed = original_seed
        p_vol_up = self.price()
        self.sigma = orig_sigma
        
        vega = p_vol_up - base_price
        
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": 0.0, "rho": 0.0}
    

    # ==========================================================================
    # NEW VISUALIZATIONS 
    # ==========================================================================

    def plot_phoenix_tunnel(self):
        """
        Visualise 200 chemins Monte Carlo colorés selon leur scénario final
        (Autocall = Vert, Maturité Safe = Gris, Perte = Rouge).
        """
        # On force 1000 simulations pour ce graph spécifique (rapide & lisible)
        original_N = self.N
        self.N = 1000 
        
        paths = self.generate_paths()
        obs_indices = self.get_observation_indices()
        
        # Logique de tri des chemins
        # Attention: paths est de dimension (Steps+1, Simulations)
        obs_prices = paths[obs_indices] # (Obs_Dates, Simulations)
        
        # Masque Autocall : Si à n'importe quelle date d'obs, Prix >= Barrière Autocall
        autocall_mask = np.any(obs_prices >= self.autocall_barrier, axis=0)
        
        final_prices = paths[-1]
        # Masque Crash : Pas autocallé ET fini sous la protection
        crash_mask = (~autocall_mask) & (final_prices < self.protection_barrier)
        # Masque Safe : Pas autocallé MAIS fini au dessus protection
        safe_mask = (~autocall_mask) & (final_prices >= self.protection_barrier)
        
        # Préparation Plotly
        fig = go.Figure()
        
        # Limite d'affichage pour ne pas surcharger le navigateur (200 lignes max)
        max_lines = 200
        x_axis = np.arange(paths.shape[0])
        
        # Fonction helper pour tracer des groupes de lignes
        def add_lines(mask, color, name, opacity):
            indices = np.where(mask)[0]
            if len(indices) == 0: return
            # On prend les 'max_lines' premiers chemins de ce groupe
            selected = indices[:max_lines]
            
            # Pour Plotly, tracer 100 lignes séparées est lourd. 
            # Astuce: On met tout dans une seule trace avec des 'None' entre les lignes
            x_flat = []
            y_flat = []
            for idx in selected:
                x_flat.extend(x_axis)
                x_flat.append(None) # Rupture de ligne
                y_flat.extend(paths[:, idx])
                y_flat.append(None)
            
            fig.add_trace(go.Scatter(
                x=x_flat, y=y_flat, 
                mode='lines', 
                line=dict(color=color, width=1), 
                opacity=opacity,
                name=name,
                showlegend=True
            ))

        # 1. Tracé des Chemins
        add_lines(autocall_mask, 'green', 'Autocall (Early Exit)', 0.15)
        add_lines(safe_mask, 'gray', 'Maturity (Capital Protected)', 0.4)
        add_lines(crash_mask, 'red', 'Loss (Barrier Hit)', 0.6)
        
        # 2. Barrières
        days = paths.shape[0] - 1
        fig.add_hline(y=self.autocall_barrier, line_dash="dash", line_color="green", annotation_text="Autocall Lvl")
        fig.add_hline(y=self.protection_barrier, line_dash="dash", line_color="red", annotation_text="Protection Lvl")
        if self.coupon_barrier != self.protection_barrier:
            fig.add_hline(y=self.coupon_barrier, line_dash="dot", line_color="cyan", annotation_text="Coupon Lvl")
            
        # 3. Dates d'observation (Lignes verticales)
        for idx in obs_indices:
            fig.add_vline(x=idx, line_width=1, line_color="white", opacity=0.2)
            
        # 4. Boite de Statistiques (Annotations)
        n_auto, n_safe, n_crash = np.sum(autocall_mask), np.sum(safe_mask), np.sum(crash_mask)
        stats_text = (
            f"<b>SCENARIOS (N={self.N})</b><br>"
            f"<span style='color:green'>Autocall: {n_auto} ({n_auto/self.N:.1%})</span><br>"
            f"<span style='color:gray'>Mature: {n_safe} ({n_safe/self.N:.1%})</span><br>"
            f"<span style='color:red'>Loss: {n_crash} ({n_crash/self.N:.1%})</span>"
        )
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.99, y=0.99,
            text=stats_text,
            showarrow=False,
            align="right",
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="white",
            borderwidth=1
        )

        fig.update_layout(
            title="Monte Carlo Path Analysis (Tunnel)",
            xaxis_title="Trading Days",
            yaxis_title="Spot Price",
            template="plotly_dark",
            showlegend=True
        )
        
        # Reset N original
        self.N = original_N
        return fig

    def plot_phoenix_distribution(self):
        """
        Histogramme de la distribution des Payoffs (actualisés).
        Montre la Value-at-Risk implicite et la moyenne (Prix).
        """
        # On utilise N simulations standard (ex: 10k ou 50k défini dans init)
        payoffs = self.calculate_payoffs_distribution()
        mean_price = np.mean(payoffs)
        
        # Conversion en % du nominal pour lecture facile
        payoffs_pct = (payoffs / self.nominal) * 100
        mean_pct = (mean_price / self.nominal) * 100
        
        fig = px.histogram(
            x=payoffs_pct, 
            nbins=60, 
            title=f"Payoff Distribution (Fair Value: {mean_pct:.2f}%)",
            color_discrete_sequence=['skyblue']
        )
        
        # Lignes verticales clés
        fig.add_vline(x=mean_pct, line_color="red", line_dash="dash", annotation_text=f"Fair Value")
        fig.add_vline(x=100, line_color="green", line_dash="dot", annotation_text="Initial Cap")

        fig.update_layout(
            xaxis_title="Payoff (% Nominal)",
            yaxis_title="Frequency",
            template="plotly_dark",
            bargap=0.1
        )
        return fig

    def plot_mc_noise_distribution(self):
        """
        Analyse de convergence : Lance 50 pricings complets pour voir la variance du prix (bruit MC).
        Sert à montrer la robustesse du prix affiché.
        """
        n_experiments = 30 # Suffisant pour la démo
        prices = []
        
        # Sauvegarde état
        original_seed = self.seed
        
        # On lance N pricings avec des seeds différents
        for i in range(n_experiments):
            self.seed = i # Change seed
            prices.append(self.price())
            
        # Reset
        self.seed = original_seed
        
        prices = np.array(prices)
        prices_pct = (prices / self.nominal) * 100
        mean = np.mean(prices_pct)
        std = np.std(prices_pct)
        
        fig = px.histogram(
            x=prices_pct,
            nbins=15,
            title=f"Monte Carlo Convergence Noise (Std Dev: {std:.2f}%)",
            color_discrete_sequence=['gray']
        )
        
        fig.add_vline(x=mean, line_color="red", line_dash="dash", annotation_text=f"Mean: {mean:.2f}%")
        
        # Zone de confiance 95%
        fig.add_vrect(
            x0=mean - 1.96*std, x1=mean + 1.96*std,
            fillcolor="yellow", opacity=0.1,
            annotation_text="95% Confidence"
        )

        fig.update_layout(
            xaxis_title="Price Estimate (% Nominal)",
            yaxis_title="Count",
            template="plotly_dark",
            bargap=0.1
        )
        return fig