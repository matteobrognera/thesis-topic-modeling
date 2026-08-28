"""
CAPITOLO 2 - PREPROCESSING TESTUALE: METODI AVANZATI
  - Metodo LLM   (via Ollama / qwen2.5:1.5b)
  - Metodo Ibrido (filtro semantico + NER/POS + frequenza + clustering)
"""

import re
import pandas as pd
import numpy as np
import gc
import ollama
from tqdm import tqdm
from collections import Counter

import spacy
from sklearn.cluster import MiniBatchKMeans
from sentence_transformers import SentenceTransformer, util


# =============================================================================
# CONFIGURAZIONE
# =============================================================================

FILE_INPUT         = "dummy_data.csv"
FILE_OUTPUT_LLM    = "output_llm.csv"
FILE_OUTPUT_HYBRID = "output_hybrid.csv"

COLONNA_INPUT  = "testo_note"
COLONNA_OUTPUT = "testo_processato"

# Parametri metodo ibrido
PROMPT_DI_INTERESSE = (
    "lavoro, mensa, doccia, cibo, senzatetto, disoccupato, ospedale, permesso di soggiorno, "
    "straniero, avvocato, dipendenza, droghe, alcol, soldi, bollette, affitto, aiuto, carcere, "
    "spesa, famiglia, ascolto"
)
SOGLIA_SIMILARITA_FRASI = 0.20
POS_VALIDI              = {"NOUN", "ADJ"}
LUNGHEZZA_MIN_LEMMA     = 3
MIN_DOCS                = 2   # abbassare a 1 se il corpus ha pochi documenti
N_CLUSTERS_RATIO        = 0.60

BLACK_LIST_NOMI = {
    "mario", "maria", "omar", "stefano", "giuliana", "fabio", "letizia",
    "vasile", "robby", "marzia"
}


# =============================================================================
# METODO 1 — LLM (Ollama + qwen2.5:1.5b)
# =============================================================================

def chunk_testo(testo, max_words=500):
    parole = testo.split()
    for i in range(0, len(parole), max_words):
        yield " ".join(parole[i:i + max_words])


def pulisci_con_llm(testo_nota):
    if not isinstance(testo_nota, str) or pd.isna(testo_nota) or not testo_nota.strip():
        return ""

    parole_totali_nota = []
    for blocco in chunk_testo(testo_nota, max_words=500):
        prompt = f"""Sei un assistente specializzato nel text mining e nella riduzione del vocabolario (text minimization) per dati sensibili in italiano.
Elabora questa porzione di testo seguendo RIGIDAMENTE queste regole:

1. ANONIMIZZAZIONE: Rimuovi qualsiasi dato sensibile (nomi, cognomi, numeri di telefono, codici fiscali, indirizzi).
2. PULIZIA LOG DI SISTEMA: Elimina completamente stringhe di log automatizzate, modifiche di sistema o pattern ripetitivi come "Modifica valore variabile X. Vecchio valore: Y". Tieni solo il testo umano.
3. CORREZIONE: Correggi eventuali errori grammaticali o refusi prima di ridurre le parole.
4. NORMALIZZAZIONE E LEMMATIZZAZIONE ESTREMA: Converti tutto in minuscolo. Riduci ogni parola rimasta alla sua forma base/radice (un mix tra stemming e lemmatizzazione, es. "pazienti" -> "paziente", "operato" -> "operare").
5. RIMOZIONE STOPWORDS E VERBI VUOTI: Elimina articoli, preposizioni, congiunzioni, pronomi e verbi ausiliari/ausiliari deboli come "essere" e "avere".
6. MINIMIZZAZIONE: Mantieni solo i termini/concetti chiave essenziali per non perdere il significato informativo della nota.

Rispondi SOLO con le parole chiave risultanti, tutte in minuscolo, separate da un singolo spazio. Non aggiungere commenti o introduzioni.

TESTO DA ELABORARE: {blocco}"""

        try:
            response = ollama.generate(
                model='qwen2.5:1.5b',
                prompt=prompt,
                options={'temperature': 0.0, 'num_predict': 250}
            )
            parole_totali_nota.extend(response['response'].strip().lower().split())
        except Exception as e:
            print(f"\nErrore durante l'elaborazione del blocco: {e}")
            continue

    return " ".join(parole_totali_nota)


def run_llm_pipeline():
    df = pd.read_csv(FILE_INPUT)
    print(f"Elaborazione di {len(df)} righe con metodo LLM...")
    note_generate = []
    for idx, nota in enumerate(tqdm(df[COLONNA_INPUT], desc="Pulizia LLM")):
        note_generate.append(pulisci_con_llm(nota))
        if idx % 10 == 0:
            gc.collect()
    df['testo_note_pulito_llm'] = note_generate
    df.to_csv(FILE_OUTPUT_LLM, index=False, encoding='utf-8')
    print(f"Output salvato in: {FILE_OUTPUT_LLM}")


# =============================================================================
# METODO 2 — IBRIDO
# (Filtro semantico + NER/POS + Frequenza documentale + Clustering embedding)
# =============================================================================

def filtra_frasi_per_rilevanza(corpus, model, target_prompt, soglia=0.20):
    print(f"\n[PASSO 1/4] Filtraggio semantico frasi (soglia={soglia})...")
    prompt_emb = model.encode(target_prompt, convert_to_tensor=True)
    corpus_filtrato = []
    for doc_text in corpus:
        frasi = [f.strip() for f in re.split(r'[.!?]\s+', doc_text) if len(f.strip()) > 10]
        if not frasi:
            corpus_filtrato.append("")
            continue
        emb = model.encode(frasi, convert_to_tensor=True, show_progress_bar=False)
        sim = util.cos_sim(emb, prompt_emb).cpu().numpy().flatten()
        corpus_filtrato.append(" ".join(frasi[i] for i, s in enumerate(sim) if s >= soglia))
    return corpus_filtrato


def pulizia_ner_e_pos(corpus, nlp_model, blacklist, pos_validi, min_len=3):
    print("\n[PASSO 2/4] Lemmatizzazione, Anti-NER e Filtro POS...")
    corpus_elaborato = []
    for doc in nlp_model.pipe(corpus, batch_size=128):
        nomi_ner = set()
        for ent in doc.ents:
            if ent.label_ == "PER":
                nomi_ner.update(ent.text.lower().split())
        parole = []
        for token in doc:
            lemma = token.lemma_.lower().strip()
            tl    = token.text.lower().strip()
            if token.is_punct or token.is_space or token.is_stop:
                continue
            if token.pos_ == "PROPN" or tl in nomi_ner or lemma in nomi_ner \
                    or tl in blacklist or lemma in blacklist:
                continue
            if token.pos_ not in pos_validi or len(lemma) < min_len:
                continue
            parole.append(lemma)
        corpus_elaborato.append(" ".join(parole))
    return corpus_elaborato


def filtra_per_frequenza_documentale(corpus, min_docs=2):
    print(f"\n[PASSO 3/4] Filtro frequenza documentale (min_docs={min_docs})...")
    doc_freq = Counter()
    for doc in corpus:
        doc_freq.update(set(doc.split()))
    valide = {w for w, c in doc_freq.items() if c >= min_docs and w.strip()}
    return [" ".join(w for w in doc.split() if w in valide) for doc in corpus]


def riduci_vocabolario_clustering(corpus, model, ratio=0.60):
    all_words = [w for w in set(" ".join(corpus).split()) if w.strip()]
    if not all_words:
        return corpus
    print(f"\n[PASSO 4/4] Clustering di {len(all_words)} lemmi con MiniLM...")
    emb = model.encode(all_words, show_progress_bar=True, batch_size=256)
    n   = max(1, min(int(len(all_words) * ratio), len(all_words)))
    labels = MiniBatchKMeans(n_clusters=n, random_state=42,
                             batch_size=min(1024, len(all_words)),
                             n_init="auto").fit_predict(emb)
    freq = Counter(" ".join(corpus).split())
    mapping = {}
    for cid in range(n):
        idx = np.where(labels == cid)[0]
        if len(idx) > 0:
            words = [all_words[i] for i in idx]
            rep   = max(words, key=lambda w: freq[w])
            for w in words:
                mapping[w] = rep
    return [" ".join(mapping.get(w, w) for w in doc.split()) for doc in corpus]


def run_hybrid_pipeline():
    print(f"\nCaricamento dati da: {FILE_INPUT}")
    df = pd.read_csv(FILE_INPUT)
    df.columns = df.columns.str.strip()
    df[COLONNA_INPUT] = df[COLONNA_INPUT].fillna("").astype(str)
    testo_originale = df[COLONNA_INPUT].tolist()

    print("Caricamento modello spaCy (it_core_news_lg)...")
    nlp = spacy.load("it_core_news_lg", disable=["parser"])

    print("Caricamento modello SentenceTransformer (all-MiniLM-L6-v2)...")
    model_embed = SentenceTransformer('all-MiniLM-L6-v2')

    print("\n================ START PIPELINE IBRIDA ================")
    t1 = filtra_frasi_per_rilevanza(testo_originale, model_embed,
                                     PROMPT_DI_INTERESSE, SOGLIA_SIMILARITA_FRASI)
    t2 = pulizia_ner_e_pos(t1, nlp, BLACK_LIST_NOMI, POS_VALIDI, LUNGHEZZA_MIN_LEMMA)
    t3 = filtra_per_frequenza_documentale(t2, min_docs=MIN_DOCS)
    t4 = riduci_vocabolario_clustering(t3, model_embed, ratio=N_CLUSTERS_RATIO)

    df[COLONNA_OUTPUT] = t4

    # Rimozione righe vuote e duplicati
    mask_vuoti     = df[COLONNA_OUTPUT].astype(str).str.strip() == ""
    mask_duplicati = df.duplicated(subset=[COLONNA_INPUT], keep='first')
    df_pulito = df[~mask_vuoti & ~mask_duplicati].copy()
    print(f"Righe iniziali: {len(df)} | Righe finali: {len(df_pulito)}")

    df_pulito.to_csv(FILE_OUTPUT_HYBRID, index=False)
    print(f"Output salvato in: {FILE_OUTPUT_HYBRID}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" METODO LLM")
    print("=" * 60)
    run_llm_pipeline()

    print("\n" + "=" * 60)
    print(" METODO IBRIDO")
    print("=" * 60)
    run_hybrid_pipeline()
