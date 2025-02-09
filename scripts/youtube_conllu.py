import os
import stanza
import multiprocessing
from tqdm import tqdm

# Percorsi delle cartelle
input_folder = "../data/clean/youtube/"
output_folder = "../data/stanza_youtube/"
num_workers = 4  # Numero di processi paralleli

# Creiamo la cartella di output se non esiste
os.makedirs(output_folder, exist_ok=True)

# Caricare il modello di Stanza per l'analisi NLP
#stanza.download("it")
nlp = stanza.Pipeline(lang="it", processors="tokenize,pos,lemma,depparse")

# Funzione per convertire un testo in formato CoNLL-U
def convert_to_conllu(text, filename):
    doc = nlp(text)
    conllu_lines = ["# newdoc", "# newpar", f"# sent_id = {filename}"]
    
    for sent in doc.sentences:
        conllu_lines.append(f"# text = {' '.join([word.text for word in sent.words])}")
        for word in sent.words:
            conllu_lines.append(
                f"{word.id}\t{word.text}\t{word.lemma}\t{word.upos}\t{word.xpos}\t_\t{word.head}\t{word.deprel}\t_\t_"
            )
        conllu_lines.append("")  # Linea vuota tra frasi
    
    return "\n".join(conllu_lines)

# Funzione per processare un singolo file
def process_file(file_path):
    filename = os.path.basename(file_path).replace(".txt", "")
    
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    
    conllu_text = convert_to_conllu(text, filename)
    output_file_path = os.path.join(output_folder, f"{filename}.conllu")
    
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(conllu_text)
    
    return filename

# Funzione principale per processare tutti i file
def process_files_in_parallel():
    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".txt")]
    
    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_file, files), total=len(files), desc="Processing TXT to CoNLL-U"))
    
    print("✅ Conversione completata! I file .conllu sono stati salvati in", output_folder)

if __name__ == "__main__":
    process_files_in_parallel()