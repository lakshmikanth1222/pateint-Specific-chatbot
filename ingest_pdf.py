import os
import io
import json
import csv
import psycopg2
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. CONFIGURATION 
# ==========================================
DB_URL = ""
DATA_FOLDER = r"C:\Users\laksh\OneDrive\Desktop\ingest\dummy_data"
CSV_PATH = r"C:\Users\laksh\Downloads\patient.csv"

# ==========================================
# 2. LOAD PATIENT METADATA FROM CSV
# ==========================================
def load_patient_map():
    print("🔄 Loading patient data from CSV...")
    patient_map = {}
    try:
        with open(CSV_PATH, mode='r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                abha_id = row.get("abha_id", "").strip()
                if abha_id.startswith("ABHA100"):
                    folder_num = abha_id.replace("ABHA100", "")
                    folder_name = f"patient_{folder_num}"
                    
                    patient_map[folder_name] = {
                        "patient_id": row["patient_id"].strip(),
                        "abha_id": abha_id,
                        "name": row.get("name", "Unknown Patient").strip()
                    }
        print(f"✅ Successfully loaded {len(patient_map)} patients from CSV.")
        return patient_map
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        exit()

# ==========================================
# 3. TEXT EXTRACTION (PyMuPDF + OCR)
# ==========================================
def extract_text(filepath):
    text = ""
    try:
        doc = fitz.open(filepath)
        for page in doc:
            text += page.get_text()

        if text.strip():
            return text

        print(f"   ⚠️ Running OCR on {os.path.basename(filepath)}...")
        for page in doc:
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            text += pytesseract.image_to_string(img)

        return text if text.strip() else "Medical report"
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return None

# ==========================================
# 4. SMART CHUNKING
# ==========================================
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    text = text.replace('\n', ' ').strip()
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# ==========================================
# 5. MAIN INGESTION PIPELINE
# ==========================================
def main():
    PATIENT_MAP = load_patient_map()

    print("🔄 Loading Embedding Model (bge-small-en-v1.5)...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    print("🔄 Connecting to Neon Database...")
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor()

    print("🔄 Verifying Database Schema matches your Neon DB...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Drops the table to prevent schema conflicts, then recreates it matching your image exactly
    cur.execute("DROP TABLE IF EXISTS patient_records;")
    cur.execute("""
        CREATE TABLE patient_records (
            id BIGSERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            embedding vector(384),
            metadata JSON
        );
    """)
    conn.commit()

    for patient_folder in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, patient_folder)
        if not os.path.isdir(path): continue

        patient = PATIENT_MAP.get(patient_folder)
        if not patient: continue

        print(f"\n➡️ Processing {patient_folder} (Name: {patient['name']}, ABHA: {patient['abha_id']})")

        for file in os.listdir(path):
            if not file.lower().endswith(".pdf"): continue

            filepath = os.path.join(path, file)
            print(f"   📄 Reading {file}...")
            text = extract_text(filepath)
            if not text: continue

            chunks = chunk_text(text)
            print(f"   ✂️ Creating {len(chunks)} vectors...")

            for chunk in chunks:
                embedding = model.encode(chunk).tolist()
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                
                metadata = {
                    "patient_id": patient["patient_id"],
                    "abha_id": patient["abha_id"],
                    "file_name": file
                }

                # Inserts using %s::json to perfectly match your database
                cur.execute(
                    """
                    INSERT INTO patient_records (text, embedding, metadata)
                    VALUES (%s, %s::vector, %s::json)
                    """,
                    (chunk, embedding_str, json.dumps(metadata))
                )

    conn.commit()
    print("\n✅ SUCCESS: All data cleanly ingested into 'patient_records' table!")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
