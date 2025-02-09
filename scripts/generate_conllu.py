'''
[ 2 ]

Ho deciso di simulare il funzionamento di profilingUD, prendendo in input i file txt dei testi e ottenendo in output i conllu corrispondenti
Dopo questo script, farò l'estrazione delle feature linguistiche dai file conllu per ottenere un csv da dare all'SVM
Alternativamente, farò anche uno script con l'uso diretto dei file csv al posto di quelli conllu, contenenti 

Lo script:
    1. Legge i file train_texts.csv e test_texts.csv, che contengono i testi.
    2. Analizza i testi con Stanza, estraendo POS tagging, dipendenze sintattiche, ecc.
    3. Salva l'output in .conllu, seguendo la struttura che userebbe anche ProfilingUD:
        # newdoc
        # newpar (all'inizio di ogni paragrafo)
        # sent_id = ... (ID del testo)

    4. Salva i file .conllu nelle cartelle:
        ../data/train_conllu/ per i testi di training.
        ../data/test_conllu/ per i testi di test.
'''

import os
import stanza
import multiprocessing
from tqdm import tqdm

# Percorsi delle cartelle
train_texts_dir = "../data/clean_texts/clean_training_texts/"
test_texts_dir = "../data/clean_texts/clean_test_texts/"
train_conllu_dir = "../data/train_conllu/"
test_conllu_dir = "../data/test_conllu/"

# Creiamo le cartelle per i file conllu se non esistono
os.makedirs(train_conllu_dir, exist_ok=True)
os.makedirs(test_conllu_dir, exist_ok=True)

# Caricare il modello di Stanza per l'analisi NLP
stanza.download("it")
nlp = stanza.Pipeline(lang="it", processors="tokenize,pos,lemma,depparse")

# Funzione per salvare il testo in formato CoNLL-U
def save_conllu(doc, filename, output_dir):
    conllu_lines = ["# newdoc", "# newpar", f"# sent_id = {filename}"]
    
    for sent in doc.sentences:
        conllu_lines.append("# newpar")  # Inizio nuovo paragrafo
        for word in sent.words:
            conllu_lines.append(f"{word.id}\t{word.text}\t{word.lemma}\t{word.upos}\t{word.xpos}\t_\t{word.head}\t{word.deprel}\t_\t_")
        conllu_lines.append("")  # Linea vuota tra frasi
    
    file_path = os.path.join(output_dir, f"{filename}.conllu")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(conllu_lines))

# Funzione per processare un singolo documento
def process_document(args):
    file_path, output_dir = args
    filename = os.path.basename(file_path).replace(".txt", "")
    
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    
    doc = nlp(text)
    save_conllu(doc, filename, output_dir)

# Funzione per processare un dataset e creare file .conllu con multiprocessing
def process_text_dataset(input_dir, conllu_dir):
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".txt")]
    
    with multiprocessing.Pool(processes=4) as pool:
        list(tqdm(pool.imap(process_document, [(f, conllu_dir) for f in files]), total=len(files), desc=f"Processing {input_dir}"))

# Creazione dei file CoNLL-U
if __name__ == "__main__":
    process_text_dataset(train_texts_dir, train_conllu_dir)
    process_text_dataset(test_texts_dir, test_conllu_dir)
    print("✅ Creazione dei file .conllu completata con multiprocessing!")





# Tempo di esecuzione per train_texts.csv: ~ 40:33 minuti (11000 testi)
# Tempo di esecuzione per test_texts.csv: ~ 30:51 minuti (10874 testi)