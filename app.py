import streamlit as pd_stream  # Streamlit per l'interfaccia web
import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime

# Configurazione della pagina Streamlit
pd_stream.set_page_config(page_title="Premier League Predictor", page_icon="⚽", layout="wide")

pd_stream.title("⚽ Premier League Statistical Predictor")
pd_stream.markdown("L'applicazione scarica i dati in tempo reale e calcola i pronostici per il prossimo turno.")

# ==============================================================================
# SORGENTI DATI & MAPPING
# ==============================================================================
url_risultati = "https://www.football-data.co.uk/mmz4281/2627/E0.csv" 
url_storico_1 = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
url_storico_2 = "https://www.football-data.co.uk/mmz4281/2425/E0.csv"

dizionario_nomi = {
    'Manchester United': 'Man United', 'Manchester City': 'Man City',
    'Newcastle United': 'Newcastle', 'Tottenham Hotspur': 'Tottenham',
    'West Ham United': 'West Ham', 'Brighton & Hove Albion': 'Brighton',
    'Wolverhampton Wanderers': 'Wolves'
}

def normalizza_nome(nome):
    return dizionario_nomi.get(nome, nome)

# ==============================================================================
# CARICAMENTO DATI (CON CACHING PER VELOCIZZARE)
# ==============================================================================
@pd_stream.cache_data(ttl=3600) # Aggiorna i dati al massimo ogni ora
def carica_dati():
    colonne = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HST', 'AST']
    
    try:
        df_curr = pd.read_csv(url_risultati)
        df_curr = df_curr[[c for c in colonne if c in df_curr.columns]]
    except:
        df_curr = pd.DataFrame(columns=colonne)
        
    dfs_hist = []
    for url in [url_storico_1, url_storico_2]:
        try:
            df_s = pd.read_csv(url)
            dfs_hist.append(df_s[[c for c in colonne if c in df_s.columns]].dropna())
        except:
            pass
            
    return df_curr, dfs_hist

df_corrente, dfs_storico = carica_dati()

# ==============================================================================
# ELABORAZIONE STATISTICA
# ==============================================================================
stats_casa, stats_trasferta = {}, {}
squadre_totali = set(df_corrente['HomeTeam'].dropna().unique()) if not df_corrente.empty else set()
for df_h in dfs_storico:
    squadre_totali.update(df_h['HomeTeam'].dropna().unique())

for squadra in squadre_totali:
    c_acc, t_acc = {'tiri': 0, 'sub': 0, 'g': 0, 'gs': 0}, {'tiri': 0, 'sub': 0, 'g': 0, 'gs': 0}
    w_c, w_t = 0, 0
    for idx, df in enumerate(dfs_storico):
        p = 1.0 if idx == 0 else 0.5
        df_c = df[df['HomeTeam'] == squadra]
        if not df_c.empty:
            w_c += p
            c_acc['tiri'] += df_c['HST'].mean() * p
            c_acc['sub'] += df_c['AST'].mean() * p
            c_acc['g'] += df_c['FTHG'].mean() * p
            c_acc['gs'] += df_c['FTAG'].mean() * p
        df_t = df[df['AwayTeam'] == squadra]
        if not df_t.empty:
            w_t += p
            t_acc['tiri'] += df_t['AST'].mean() * p
            t_acc['sub'] += df_t['HST'].mean() * p
            t_acc['g'] += df_t['FTAG'].mean() * p
            t_acc['gs'] += df_t['FTHG'].mean() * p
            
    if w_c > 0: stats_casa[squadra] = {k: v/w_c for k, v in c_acc.items()}
    if w_t > 0: stats_trasferta[squadra] = {k: v/w_t for k, v in t_acc.items()}

df_prof_c = pd.DataFrame(stats_casa).T
df_prof_t = pd.DataFrame(stats_trasferta).T

# Estrazione palinsesto automatico dalle righe senza risultato
prossimo_turno = []
if not df_corrente.empty:
    df_future = df_corrente[df_corrente['FTHG'].isna() | df_corrente['FTHG'].isnull()]
    if not df_future.empty:
        for _, riga in df_future.head(10).iterrows():
            prossimo_turno.append((riga['HomeTeam'], riga['AwayTeam']))

# Fallback se siamo in pre-stagione (es. Luglio)
if not prossimo_turno:
    prossimo_turno = [('Arsenal', 'Chelsea'), ('Man City', 'Liverpool'), ('Man United', 'Tottenham')]

# ==============================================================================
# INTERFACCIA UTENTE
# ==============================================================================
st_hst_default = 4.5
st_sub_default = 5.2

if pd_stream.button("🔄 Forza Aggiornamento Dati"):
    pd_stream.cache_data.clear()
    pd_stream.rerun()

elenco_report = []

for casa, trasferta in prossimo_turno:
    st_hst_c = df_prof_c.loc[casa, 'tiri'] if casa in df_prof_c.index else st_hst_default
    st_sub_t = df_prof_t.loc[trasferta, 'sub'] if trasferta in df_prof_t.index else st_sub_default
    st_gol_c = df_prof_c.loc[casa, 'g'] if casa in df_prof_c.index else 1.2
    st_subg_t = df_prof_t.loc[trasferta, 'gs'] if trasferta in df_prof_t.index else 1.6
    
    st_hst_t = df_prof_t.loc[trasferta, 'tiri'] if trasferta in df_prof_t.index else 3.8
    st_sub_c = df_prof_c.loc[casa, 'sub'] if casa in df_prof_c.index else st_hst_default
    st_gol_t = df_prof_t.loc[trasferta, 'g'] if trasferta in df_prof_t.index else 1.0
    st_subg_c = df_prof_c.loc[casa, 'gs'] if casa in df_prof_c.index else 1.5

    forma_c = df_corrente[df_corrente['HomeTeam'] == casa].tail(5) if not df_corrente.empty else pd.DataFrame()
    forma_t = df_corrente[df_corrente['AwayTeam'] == trasferta].tail(5) if not df_corrente.empty else pd.DataFrame()
    
    if len(forma_c) >= 3:
        tiri_attesi_c = (forma_c['HST'].mean() * 0.25 + st_hst_c * 0.75 + st_sub_t) / 2
        gol_attesi_c = forma_c['FTHG'].mean() * 0.25 + st_gol_c * 0.75
    else:
        tiri_attesi_c = (st_hst_c + st_sub_t) / 2
        gol_attesi_c = st_gol_c
        
    if len(forma_t) >= 3:
        tiri_attesi_t = (forma_t['AST'].mean() * 0.25 + st_hst_t * 0.75 + st_sub_c) / 2
        gol_attesi_t = forma_t['FTAG'].mean() * 0.25 + st_gol_t * 0.75
    else:
        tiri_attesi_t = (st_hst_t + st_sub_c) / 2
        gol_attesi_t = st_gol_t

    tiri_totali = tiri_attesi_c + tiri_attesi_t
    
    lambda_h = max(0.2, (gol_attesi_c * st_subg_t) / 1.60)
    lambda_a = max(0.2, (gol_attesi_t * st_subg_c) / 1.30)
    prob_btts = (1 - stats.poisson.pmf(0, lambda_h)) * (1 - stats.poisson.pmf(0, lambda_a))
    quota_minima_goal = 1 / max(prob_btts, 0.01) * 1.10
    
    segno_over = "🟢 GIOCARE OVER 2.5" if tiri_totali > 9.5 else "❌ NO OVER"
    
    elenco_report.append({
        "Partita": f"🏟️ {casa} vs {trasferta}",
        "Tiri in Porta Stimati": round(tiri_totali, 2),
        "Consiglio Over 2.5": segno_over,
        "Quota Minima GOAL": f"> {round(quota_minima_goal, 2)}"
    })

# Mostra i dati in una tabella elegante all'interno della pagina web
df_report = pd.DataFrame(elenco_report)
pd_stream.dataframe(df_report, use_container_width=True, hide_index=True)

pd_stream.caption(f"Ultimo aggiornamento effettuato il: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
