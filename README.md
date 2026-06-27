# English to Hindi Translation Pipeline

This repository contains the work for Assignment 1 (Dataset Processing) and Assignment 2 (Translation Inference).

## Files Included
* **`translation_inference.py`**: The main Python script that runs inference using the `facebook/nllb-200-distilled-600M` model.
* **`assignment1_cleaned.xlsx`**: The cleaned dataset containing over 10,000 sentences filtered by length (3 to 60 words).
* **`assessment2_translations.xlsx`**: The final output file containing the 200 original English sentences and the model-generated Hindi translations.

## How to Run the Code
1. Clone this repository to your local system:
   `git clone https://github.com/Divine-Emperor/Harsh_NK9.git`
2. Install the required dependencies:
   `pip install torch pandas sacrebleu transformers openpyxl`
3. Ensure `assignment1_cleaned.xlsx` is in the same directory as the script.
4. Run the script:
   `python translation_inference.py`

## Evaluation Scores
The model outputs were compared against the original Hindi ground truth references using `sacrebleu`. 

* **BLEU Score:** 0.9977
* **CHRF Score:** 12.6159
* **TER Score:** 189.2542
