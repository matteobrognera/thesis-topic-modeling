# =============================================================================
# CAPITOLO 2 - PREPROCESSING TESTUALE: METODO STANDARD
# Pulizia, Stemming e Lemmatizzazione
# =============================================================================

library(dplyr)
library(tm)
library(SnowballC)
library(udpipe)

# -----------------------------------------------------------------------------
# 1. CARICAMENTO DATASET
# -----------------------------------------------------------------------------
df <- read.csv("dummy_data.csv", stringsAsFactors = FALSE)

df_note <- df %>%
  filter(n_note != 0) %>%
  filter(!is.na(testo_note) & trimws(testo_note) != "")

# -----------------------------------------------------------------------------
# 2. PULIZIA TESTUALE DI BASE
# -----------------------------------------------------------------------------
corpus_raw <- Corpus(VectorSource(df_note$testo_note))

corpus_clean <- corpus_raw %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("italian")) %>%
  tm_map(stripWhitespace)

# -----------------------------------------------------------------------------
# 3. STEMMING
# -----------------------------------------------------------------------------
corpus_stemmed <- tm_map(corpus_clean, stemDocument, language = "italian")
df_note$testo_stemmed <- sapply(corpus_stemmed, as.character)

# -----------------------------------------------------------------------------
# 4. LEMMATIZZAZIONE (via udpipe)
# -----------------------------------------------------------------------------

# Scarica il modello la prima volta (poi commentare la riga seguente)
# udpipe_download_model(language = "italian")
udmodel <- udpipe_load_model("italian-isdt-ud-2.5-191206.udpipe")

lemmati    <- udpipe_annotate(udmodel, x = df_note$testo_note,
                              doc_id = as.character(df_note$id_anag))
lemmati_df <- as.data.frame(lemmati)

lemmi_per_doc <- lemmati_df %>%
  filter(upos %in% c("NOUN", "ADJ")) %>%
  filter(!is.na(lemma) & nchar(lemma) >= 3) %>%
  group_by(doc_id) %>%
  summarise(testo_lemmatizzato = paste(lemma, collapse = " "), .groups = "drop")

df_note <- df_note %>%
  mutate(id_anag_chr = as.character(id_anag)) %>%
  left_join(lemmi_per_doc, by = c("id_anag_chr" = "doc_id")) %>%
  select(-id_anag_chr)

# -----------------------------------------------------------------------------
# 5. COSTRUZIONE DELLA DTM (da testo lemmatizzato)
# -----------------------------------------------------------------------------
corpus_lemma <- Corpus(VectorSource(df_note$testo_lemmatizzato))
dtm <- DocumentTermMatrix(corpus_lemma)

row_totals <- apply(dtm, 1, sum)
dtm        <- dtm[row_totals > 0, ]

cat("DTM costruita:", nrow(dtm), "documenti x", ncol(dtm), "termini\n")
