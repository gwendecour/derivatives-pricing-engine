import yfinance as yf
import numpy as np
import pandas as pd
import requests
import re
import json

# ==============================================================================
# DONNÉES DE SECOURS (BACKUP)
# Intégrées directement dans le module pour garantir une disponibilité 100%
# ==============================================================================
# ==============================================================================
# DONNÉES DE SECOURS (BACKUP)
# Intégrées directement dans le module pour garantir une disponibilité 100%
# ==============================================================================
BACKUP_JSON_DATA = """
{
  "extended": [
    {
      "maturityDate": "Dec 2026",
      "rowc": [
        {"strike": "55.00", "best_bid": "16.39", "best_ask": "16.68", "last": "16.50", "atTheMoney": false},
        {"strike": "60.00", "best_bid": "12.95", "best_ask": "13.23", "last": "13.00", "atTheMoney": false},
        {"strike": "70.00", "best_bid": "7.56", "best_ask": "7.79", "last": "7.65", "atTheMoney": true},
        {"strike": "75.00", "best_bid": "5.61", "best_ask": "5.81", "last": "5.70", "atTheMoney": false},
        {"strike": "80.00", "best_bid": "4.09", "best_ask": "4.29", "last": "4.20", "atTheMoney": false},
        {"strike": "90.00", "best_bid": "2.13", "best_ask": "2.31", "last": "2.20", "atTheMoney": false},
        {"strike": "100.00", "best_bid": "1.11", "best_ask": "1.23", "last": "1.15", "atTheMoney": false}
      ],
      "rowp": [
        {"strike": "55.00", "best_bid": "3.58", "best_ask": "3.75", "last": "3.65", "atTheMoney": false},
        {"strike": "60.00", "best_bid": "5.17", "best_ask": "5.34", "last": "5.25", "atTheMoney": false},
        {"strike": "70.00", "best_bid": "9.69", "best_ask": "9.89", "last": "9.80", "atTheMoney": true},
        {"strike": "75.00", "best_bid": "12.66", "best_ask": "12.87", "last": "12.75", "atTheMoney": false},
        {"strike": "80.00", "best_bid": "16.08", "best_ask": "16.30", "last": "16.20", "atTheMoney": false},
        {"strike": "90.00", "best_bid": "23.99", "best_ask": "24.26", "last": "24.10", "atTheMoney": false},
        {"strike": "100.00", "best_bid": "32.42", "best_ask": "33.42", "last": "33.00", "atTheMoney": false}
      ]
    }
  ]
}
"""

class MarketData:
    @staticmethod
    def get_spot(ticker):
        try:
            stock = yf.Ticker(ticker)
            # Fast info est souvent plus robuste pour le temps réel
            try:
                price = stock.fast_info.last_price
            except:
                price = None
                
            if price is None:
                price = stock.history(period="1d")['Close'].iloc[-1]
            return price
        except Exception as e:
            print(f"Error fetching Spot: {e}. Defaulting to 100.")
            return 100.0

    @staticmethod
    def get_volatility(ticker, window="1y"):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=window)
            hist['Log_Ret'] = np.log(hist['Close'] / hist['Close'].shift(1))
            annualized_vol = hist['Log_Ret'].std() * np.sqrt(252)
            return annualized_vol
        except Exception as e:
            return 0.20

    @staticmethod
    def get_dividend_yield(ticker):
        try:
            stock = yf.Ticker(ticker)
            div_yield = stock.info.get('dividendYield', 0.0)
            if div_yield is None:
                return 0.0
            return div_yield 
        except Exception as e:
            return 0.0
        
    @staticmethod
    def get_risk_free_rate(ticker="^TNX"):
        try:
            bond = yf.Ticker(ticker)
            yield_index = bond.history(period="1d")['Close'].iloc[-1]
            return yield_index / 100
        except Exception as e:
            return 0.03
    
    @staticmethod
    def get_euronext_chain(root_ticker="GL1", exchange="DPAR", maturity="12-2026"):
        """
        Récupère la chaîne d'options. 
        Tente d'abord Euronext Live, et bascule sur le BACKUP JSON si échec.
        """
        url = f"https://live.euronext.com/en/ajax/getPricesOptionsAjax/stock-options/{root_ticker}/{exchange}"
        
        # Headers "Naviguateur Réel" pour éviter le blocage 403
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": f"https://live.euronext.com/en/product/stock-options/{root_ticker}-{exchange}",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest"
        }
        
        payload = {
            "md[]": maturity,
            "revol": "yes"
        }

        print(f"--- Connecting to Euronext ({root_ticker} / {maturity}) ---")
        
        # 1. TENTATIVE DE CONNEXION
        json_data = None
        using_backup = False
        
        try:
            response = requests.post(url, data=payload, headers=headers, timeout=5)
            if response.status_code == 200:
                json_data = response.json()
                # Vérification que le JSON contient bien des données valides
                if not json_data or 'extended' not in json_data or not json_data['extended'] or json_data['extended'][0] is None:
                     raise ValueError("JSON vide ou invalide reçu d'Euronext")
            else:
                raise ConnectionError(f"Status Code: {response.status_code}")
                
        except Exception as e:
            print(f">> ÉCHEC CONNEXION EURONEXT ({e}).")
            print(">> ACTIVATION DU MODE BACKUP (Données statiques).")
            json_data = json.loads(BACKUP_JSON_DATA)
            using_backup = True

        # 2. PARSING DES DONNÉES (Valable pour Live et Backup)
        try:
            data_by_strike = {}

            def clean_price(val):
                if isinstance(val, (int, float)): return float(val)
                if not val or not isinstance(val, str) or val in ['-', 'None', '']: return np.nan
                # Nettoyage HTML si présent
                text = re.sub(r'<[^>]+>', '', val).replace(',', '').strip()
                return float(text) if text else np.nan

            def extract_strike(val):
                if isinstance(val, (int, float)): return float(val)
                # Si c'est du HTML <a...>55.00</a>
                if '<' in str(val):
                    match = re.search(r'>([\d\.,]+)<', str(val))
                    if match:
                        return float(match.group(1).replace(',', ''))
                # Si c'est juste du texte "55.00"
                return clean_price(val)

            # Parcours des blocs (au cas où Euronext change l'ordre)
            found_data = False
            if 'extended' in json_data:
                for block in json_data['extended']:
                    if not block: continue
                    
                    found_data = True
                    
                    # Calls
                    for item in block.get('rowc', []):
                        k = extract_strike(item.get('strike'))
                        if not k or np.isnan(k): continue
                        
                        if k not in data_by_strike: data_by_strike[k] = {}
                        data_by_strike[k]['Call_Bid'] = clean_price(item.get('best_bid'))
                        data_by_strike[k]['Call_Ask'] = clean_price(item.get('best_ask'))
                        data_by_strike[k]['Call_Last'] = clean_price(item.get('last'))

                    # Puts (le backup n'a pas forcément de puts, mais le live oui)
                    for item in block.get('rowp', []):
                        k = extract_strike(item.get('strike'))
                        if not k or np.isnan(k): continue
                        
                        if k not in data_by_strike: data_by_strike[k] = {}
                        data_by_strike[k]['Put_Bid'] = clean_price(item.get('best_bid'))
                        data_by_strike[k]['Put_Ask'] = clean_price(item.get('best_ask'))
                        data_by_strike[k]['Put_Last'] = clean_price(item.get('last'))

            if not found_data and not using_backup:
                # Si le live n'a rien donné malgré un JSON valide (rare), on force le backup
                print(">> Données Live vides. Force Backup.")
                json_data = json.loads(BACKUP_JSON_DATA)
                # On relance un parsing rapide du backup (récursivité simplifiée)
                # ... (code simplifié : on suppose que le backup fonctionne au prochain appel ou on renvoie vide)
                # Pour éviter la complexité, on retourne le dataframe vide ici si le backup n'a pas été chargé au début
                return pd.DataFrame()

            # 3. FORMATAGE DATAFRAME
            rows = []
            for strike, data in data_by_strike.items():
                row = {'Strike': strike, 'Maturity': maturity}
                row.update(data)
                rows.append(row)
            
            df = pd.DataFrame(rows)
            if df.empty:
                return pd.DataFrame()

            df = df.sort_values('Strike').reset_index(drop=True)
            
            # Compléter les colonnes manquantes (ex: Put si on est en Backup Calls only)
            cols_needed = ['Call_Bid', 'Call_Ask', 'Call_Last', 'Put_Bid', 'Put_Ask', 'Put_Last']
            for col in cols_needed:
                if col not in df.columns:
                    df[col] = np.nan
            
            print(f"Success! Retrieved {len(df)} strikes ({'BACKUP' if using_backup else 'LIVE'}).")
            return df[['Strike', 'Maturity', 'Call_Bid', 'Call_Ask', 'Call_Last', 'Put_Bid', 'Put_Ask', 'Put_Last']]
            
        except Exception as e:
            print(f"Parsing Failed: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_historical_data(ticker, start_date, end_date):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            if df.empty: return None
            return df['Close']
        except Exception as e:
            return None