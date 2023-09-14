from transformers import MarianMTModel, MarianTokenizer

# python manage.py

languages = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "chinese": "zh",
    "russian": "ru",
    "arabic": "ar",
    "japanese": "ja",
    "italian": "it",
    "portuguese": "pt",
    "dutch": "nl",
    "korean": "ko",
    "turkish": "tr"
    # Add more languages
}

# Replace MODEL_NAME with the pre-trained model name for your desired language pair
model_name = "Helsinki-NLP/opus-mt-es-en"  # For example, "Helsinki-NLP/opus-mt-en-es" for English to Spanish

# Load the pre-trained model and tokenizer
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)


def translate_text(text, model, tokenizer):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Perform the translation
    outputs = model.generate(**inputs)

    # Decode the translated output
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translated_text

# Example usage:
input_text = "hola mi color favorito es el rojo"
translated_text = translate_text(input_text, model, tokenizer)
print(f"Translated: {translated_text}")
