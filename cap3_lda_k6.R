# =============================================================================
# CAPITOLO 3 - MODELLO LDA DEFINITIVO CON K = 6
# =============================================================================

library(tm)
library(topicmodels)

# -----------------------------------------------------------------------------
# 1. CARICAMENTO DATASET E COSTRUZIONE DTM
# -----------------------------------------------------------------------------
df_finale <- read.csv("dummy_data.csv", stringsAsFactors = FALSE)

corpus <- Corpus(VectorSource(df_finale$testo_processato))
dtm    <- DocumentTermMatrix(corpus)

# Rimozione documenti vuoti
row_totals <- apply(dtm, 1, sum)
dtm        <- dtm[row_totals > 0, ]

cat("DTM:", nrow(dtm), "documenti x", ncol(dtm), "termini\n")

# -----------------------------------------------------------------------------
# 2. STIMA DEL MODELLO LDA CON K = 6 (METODO GIBBS)
# -----------------------------------------------------------------------------
set.seed(1234)

lda_model6 <- LDA(
  dtm,
  k      = 6,
  method = "Gibbs",
  control = list(
    iter   = 10000,
    burnin = 1000,
    alpha  = 0.1,
    delta  = 0.001
  )
)

# -----------------------------------------------------------------------------
# 3. RISULTATI: TOP PAROLE PER TOPIC
# -----------------------------------------------------------------------------
top_n      <- min(20, ncol(dtm))
top_parole <- terms(lda_model6, top_n)
colnames(top_parole) <- paste0("Topic_", 1:6)

print("--- Top parole per topic (K = 6) ---")
print(top_parole)
