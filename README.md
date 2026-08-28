# Thesis Repository — Text Mining su Dati Assistenziali

Questo repository contiene il codice riprodotto dalla tesi di laurea magistrale.  
I dati reali sono riservati: tutti gli script leggono il file `dummy_data.csv`, un dataset fittizio con la stessa struttura dei dati originali.

---

## Struttura

```
github_repository/
├── dummy_data.csv                  # Dataset di prova (dati fittizi)
├── cap2_preprocessing_standard.R   # Cap. 2 — Pulizia standard, stemming, lemmatizzazione (R)
├── cap2_preprocessing_avanzato.py  # Cap. 2 — Metodo LLM e metodo ibrido (Python)
├── cap3_lda_k6.R                   # Cap. 3 — Modello LDA con K = 6 (R)
└── cap4_hdp.py                     # Cap. 4 — Modello HDP con tomotopy (Python)
```

---

## Dipendenze

### R
```r
install.packages(c("tm", "SnowballC", "udpipe", "topicmodels",
                   "textmineR", "Matrix", "dplyr", "tidyr", "rlang", "tibble"))
```

### Python
```bash
pip install pandas numpy tomotopy spacy sentence-transformers scikit-learn scipy ollama tqdm
python -m spacy download it_core_news_lg
```

> **Nota**: il metodo LLM richiede [Ollama](https://ollama.com) installato localmente con il modello `qwen2.5:1.5b`.

---

## Ordine di esecuzione consigliato

1. `cap2_preprocessing_standard.R` — produce il testo lemmatizzato in R  
2. `cap2_preprocessing_avanzato.py` — produce `output_llm.csv` e `output_hybrid.csv`  
3. `cap3_lda_k6.R` — stima LDA su `dummy_data.csv` (colonna `testo_processato`)  
4. `cap4_hdp.py` — stima HDP e produce `hdp_topics_by_doc.csv`
