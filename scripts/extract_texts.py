'''
Creo una cartella per ciascun genere testuale e ci metto i testi (dopo averli puliti) presi dai file .txt originali, un .txt per ogni <doc>
'''


import os
import re
from tqdm import tqdm  # Per la barra di progresso

# Percorsi delle cartelle (corretti per esecuzione da 'scripts')
train_folder = "../data/original/training/"
test_folder = "../data/original/test/"
clean_texts_dir = "../data/clean/"

# Generi testuali
genres = ["children", "diary", "journalism", "twitter", "youtube"]

# Creiamo le cartelle per i generi se non esistono
for genre in genres:
    os.makedirs(os.path.join(clean_texts_dir, genre), exist_ok=True)

# Funzione per pulire il testo
def clean_text(text):
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Rimuove URL, inclusi quelli con "www."
    text = re.sub(r'(?<=\s)amzn\.com\S+', '', text)  # Rimuove link specifici di amzn.com
    text = re.sub(r'@\w+', '', text)  # Rimuove menzioni (@user)
    text = re.sub(r'#\w+', '', text)  # Rimuove hashtag
    text = re.sub(r'\b\w{10,}\b', '', text)  # Rimuove parole molto lunghe (es. codici random)
    text = re.sub(r'[^\w\s,.!?;]', '', text)  # Rimuove caratteri speciali eccetto punteggiatura
    text = re.sub(r'\s+', ' ', text).strip()  # Rimuove spazi multipli
    return text

# Funzione per estrarre i dati dai file .txt
def extract_texts_from_file(filepath, clean_dir, dataset_type):
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()

    # Regex per catturare le informazioni nel tag <doc>
    pattern = re.compile(r'<doc id="(\d+)" genre="(.*?)" gender="(.*?)">(.*?)</doc>', re.DOTALL)

    for match in pattern.finditer(content):
        doc_id, genre, gender, text = match.groups()
        text = text.strip().replace("\n", " ")  # Pulizia del testo da spazi iniziali e finali e newline

        # Pulizia del testo
        cleaned_text = clean_text(text)

        # Creazione del nome file
        gender_safe = gender if gender in ["M", "F"] else "unknown"
        filename = f"{dataset_type}#{doc_id}#{genre}#{gender_safe}.txt"
        
        # Salvataggio nella cartella del genere
        genre_dir = os.path.join(clean_texts_dir, genre)
        file_path = os.path.join(genre_dir, filename)
        
        with open(file_path, "w", encoding="utf-8") as clean_text_file:
            clean_text_file.write(cleaned_text)

# Funzione per processare i dati
def process_data(input_folder, clean_dir, dataset_type):
    if not os.path.exists(input_folder):
        print(f"⚠️ La cartella {input_folder} non esiste! Creala e inserisci i file di testo.")
        return

    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".txt")]

    if not files:
        print(f"⚠️ Nessun file .txt trovato in {input_folder}!")
        return

    for filepath in tqdm(files, desc=f"Processing {dataset_type}"):
        extract_texts_from_file(filepath, clean_dir, dataset_type)

# Anche se non uso il multiprocessing qui, metto il controllo per evitare errori
if __name__ == "__main__":
    # Elaborazione file di training e test
    process_data(train_folder, clean_texts_dir, "training")
    process_data(test_folder, clean_texts_dir, "test")

    print("✅ Testi puliti salvati in clean/{genere}/ per ciascun genere!")