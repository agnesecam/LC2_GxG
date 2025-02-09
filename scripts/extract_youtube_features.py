import os
import pandas as pd
from tqdm import tqdm
import stanza

# Percorsi delle cartelle
input_folder = "../data/profiling_output/youtube/linguistic_annotation/"
output_csv = "../data/youtube/linguistic_profile.csv"
num_workers = 4  # Numero di processi paralleli

# Caricare il modello di Stanza per l'analisi NLP
# stanza.download("it")
nlp = stanza.Pipeline(lang="it", processors="tokenize,pos,lemma,depparse")

# Definire le colonne del CSV
columns = [
    "Filename", "n_sentences", "n_tokens", "tokens_per_sent", "char_per_tok", "ttr_lemma_chunks_100", "ttr_lemma_chunks_200", 
    "ttr_form_chunks_100", "ttr_form_chunks_200", "upos_dist_ADJ", "upos_dist_ADP", "upos_dist_ADV", "upos_dist_AUX", 
    "upos_dist_CCONJ", "upos_dist_DET", "upos_dist_INTJ", "upos_dist_NOUN", "upos_dist_NUM", "upos_dist_PRON", "upos_dist_PROPN", 
    "upos_dist_PUNCT", "upos_dist_SCONJ", "upos_dist_SYM", "upos_dist_VERB", "upos_dist_X", "lexical_density"
]

# Funzione per estrarre feature linguistiche da un file .conllu
def extract_features_from_conllu(conllu_path):
    with open(conllu_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    n_sentences = sum(1 for line in lines if line.startswith("# sent_id"))
    n_tokens = sum(1 for line in lines if line and not line.startswith("#") and not line.isspace())
    
    tokens_per_sent = n_tokens / n_sentences if n_sentences > 0 else 0
    char_per_tok = sum(len(line.split("\t")[1]) for line in lines if "\t" in line) / n_tokens if n_tokens > 0 else 0
    
    upos_counts = {upos: 0 for upos in ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]}
    total_upos = 0
    
    for line in lines:
        if "\t" in line:
            fields = line.split("\t")
            upos = fields[3]
            if upos in upos_counts:
                upos_counts[upos] += 1
                total_upos += 1
    
    upos_ratios = {key: value / total_upos if total_upos > 0 else 0 for key, value in upos_counts.items()}
    lexical_density = (upos_counts["NOUN"] + upos_counts["VERB"] + upos_counts["ADJ"] + upos_counts["ADV"]) / n_tokens if n_tokens > 0 else 0
    
    # Creiamo la riga con tutte le feature estratte
    row = [os.path.basename(conllu_path), n_sentences, n_tokens, tokens_per_sent, char_per_tok] + list(upos_ratios.values()) + [lexical_density]
    
    # Debug: Stampiamo se il numero di colonne non corrisponde
    if len(row) != len(columns):
        print(f"⚠️ Mismatch colonne per {conllu_path}: previsto {len(columns)}, trovato {len(row)}")
        row += [0] * (len(columns) - len(row))  # Riempie con 0 se mancano colonne
    
    return row

# Funzione per processare i file .conllu
def process_conllu_dataset():
    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".conllu")]
    data = []
    
    for conllu_path in tqdm(files, desc="Processing CoNLL-U files"):
        data.append(extract_features_from_conllu(conllu_path))
    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ File CSV generato: {output_csv}")

if __name__ == "__main__":
    process_conllu_dataset()