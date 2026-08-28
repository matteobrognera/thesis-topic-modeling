"""
CAPITOLO 4 - MODELLO HDP (Hierarchical Dirichlet Process)
Implementazione con tomotopy
"""

import re
import pandas as pd
import tomotopy as tp


# =============================================================================
# 1. CARICAMENTO DATASET
# =============================================================================

df = pd.read_csv("dummy_data.csv")
print(f"Dataset caricato. Righe totali: {len(df)}")


# =============================================================================
# 2. TOKENIZZAZIONE
# =============================================================================

def estrai_token(testo):
    if pd.isna(testo):
        return []
    return re.findall(r'\b\w{2,}\b', str(testo).lower())

documenti_tokenizzati = [estrai_token(t) for t in df['testo_processato']]

tutti_i_token = [tok for doc in documenti_tokenizzati for tok in doc]
print(f"Parole totali nel corpus  : {len(tutti_i_token)}")
print(f"Parole uniche (dizionario): {len(set(tutti_i_token))}")


# =============================================================================
# 3. INIZIALIZZAZIONE MODELLO HDP
# =============================================================================
# alpha : iperparametro di concentrazione locale  (livello documento)
# gamma : iperparametro di concentrazione globale (livello corpus)
# eta   : iperparametro di Dirichlet per le parole nei topic
# seed  : garantisce la riproducibilità dei risultati

mdl = tp.HDPModel(
    tw     = tp.TermWeight.ONE,
    alpha  = 0.1,
    gamma  = 1.0,
    eta    = 0.01,
    seed   = 42,
    min_df = 1,    # abbassare a 1 per corpus piccoli
    rm_top = 0
)

mdl.optim_interval = 10  # ottimizzazione automatica iperparametri ogni 10 iterazioni

for doc in documenti_tokenizzati:
    if len(doc) > 0:
        mdl.add_doc(doc)

print("Modello HDP inizializzato e documenti caricati.")


# =============================================================================
# 4. ADDESTRAMENTO
# =============================================================================

print("Inizio addestramento...")
mdl.burn_in = 1000

for i in range(0, 10000, 100):
    mdl.train(100, workers=1)
    print(
        f"Iterazione {i+100:4d} | "
        f"Log-Likelihood: {mdl.ll_per_word:.4f} | "
        f"Topic Vivi: {mdl.k:3d} | "
        f"Alpha: {mdl.alpha:.4f} | "
        f"Gamma: {mdl.gamma:.4f}"
    )


# =============================================================================
# 5. RISULTATI: TOP PAROLE PER TOPIC VIVI
# =============================================================================

print("\n--- TOPIC ESTRATTI DAL MODELLO HDP ---")

top_n          = min(20, mdl.num_vocabs)
live_topics    = [k for k in range(mdl.k) if mdl.is_live_topic(k)]
conteggi_topic = mdl.get_count_by_topics()

for k in live_topics:
    parole_chiave = mdl.get_topic_words(k, top_n=top_n)
    termini_fmt   = [f"{word} ({prob:.3f})" for word, prob in parole_chiave]
    print(f"\nTopic #{k} | Token assegnati: {conteggi_topic[k]}")
    print(" -> " + ", ".join(termini_fmt))

print(f"\nTopic vivi totali: {len(live_topics)}")
