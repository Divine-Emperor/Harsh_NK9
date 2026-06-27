import torch
import pandas as pd
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. SETUP AND CONFIGURATION
input_excel_file = "assignment1_cleaned.xlsx" 
output_excel_file = "assessment2_translations.xlsx"
scores_text_file = "evaluation_scores.txt"

print("Step 1: Loading your Assignment 1 dataset...")
try:
    df_all = pd.read_excel(input_excel_file)
except FileNotFoundError:
    print(f"Error: Could not find '{input_excel_file}'. Make sure it is in the same folder as this script.")
    exit()

# Take exactly the first 200 rows
df_200 = df_all.head(200).copy()

# Select columns by position (Index 0 = English, Index 1 = Hindi) to avoid Header KeyErrors
english_sentences = df_200.iloc[:, 0].astype(str).tolist()
original_hindi_references = df_200.iloc[:, 1].astype(str).tolist()

print(f"Loaded {len(english_sentences)} sentences successfully.")

# 2. INITIALIZE HUGGING FACE MODEL (NLLB-200)
print("\nStep 2: Initializing Facebook's NLLB-200 translation model...")
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Auto-detect hardware
local_device = "cuda" if torch.cuda.is_available() else "cpu"
if local_device == "cuda":
    print("GPU detected! Translation will be fast.")
else:
    print("No GPU detected. Using CPU (This will take a few minutes...)")

# Move the model to the correct hardware
model = model.to(local_device)

# Configure languages
tokenizer.src_lang = "eng_Latn"
hindi_lang_id = tokenizer.convert_tokens_to_ids("hin_Deva")
# 3. RUN LLM INFERENCE NATIVELY
print("\nStep 3: Translating 200 sentences into Hindi...")
model_generated_hindi = []

for idx, sentence in enumerate(english_sentences):
    # Prepare text and send to GPU/CPU
    inputs = tokenizer(sentence, return_tensors="pt").to(local_device)
    
    # Generate translation
    translated_tokens = model.generate(
        **inputs, 
        forced_bos_token_id=hindi_lang_id, 
        max_length=512
    )
    
    # Decode text
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    model_generated_hindi.append(translated_text)
    
    if (idx + 1) % 20 == 0:
        print(f"Translated {idx + 1}/200 sentences...")

print("Translation completed.")

# 4. CALCULATE METRICS (BLEU, CHRF, TER)
print("\nStep 4: Computing translation accuracy metrics...")
references_list = [original_hindi_references]

bleu_score = sacrebleu.corpus_bleu(model_generated_hindi, references_list).score
chrf_score = sacrebleu.corpus_chrf(model_generated_hindi, references_list).score
ter_score = sacrebleu.corpus_ter(model_generated_hindi, references_list).score

print(f"--- Results ---")
print(f"BLEU Score: {bleu_score:.4f}")
print(f"CHRF Score: {chrf_score:.4f}")
print(f"TER Score:  {ter_score:.4f}")

# Save scores
with open(scores_text_file, "w", encoding="utf-8") as f:
    f.write("=== Evaluation Scores ===\n")
    f.write(f"BLEU Score: {bleu_score:.4f}\n")
    f.write(f"CHRF Score: {chrf_score:.4f}\n")
    f.write(f"TER Score:  {ter_score:.4f}\n")

# 5. PREPARE EXCEL OUTPUT
print("\nStep 5: Exporting final translations to Excel...")
output_df = pd.DataFrame({
    "Original English sentence": english_sentences,
    "Model-generated Hindi translation": model_generated_hindi
})

output_df.to_excel(output_excel_file, index=False)
print(f"Saved outputs! Ready for GitHub.")