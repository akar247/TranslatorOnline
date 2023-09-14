from django.shortcuts import render
from .forms import TranslationForm
from .translator import translate_text

from transformers import MarianMTModel, MarianTokenizer


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

def get_model_name(src_lang, tgt_lang):
    return f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

def translate_text(text, model, tokenizer):
    try:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Perform the translation
        outputs = model.generate(**inputs)

        # Decode the translated output
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return translated_text

    except Exception as e:
        return f"Error: {str(e)}"

def translation_view(request):
    if request.method == 'POST':
        form = TranslationForm(request.POST)
        if form.is_valid():
            src_lang = form.cleaned_data['original_language'].lower()
            tgt_lang = form.cleaned_data['target_language'].lower()
            input_text = form.cleaned_data['input_text']

            # Check if the original or target language is English
            if src_lang == 'english' or tgt_lang == 'english':
                # Perform normal translation
                model_name = get_model_name(languages[src_lang], languages[tgt_lang])
                model = MarianMTModel.from_pretrained(model_name)
                tokenizer = MarianTokenizer.from_pretrained(model_name)

                translated_text = translate_text(input_text, model, tokenizer)
            else:
                # Translate from original language to English first,
                # then from English to target language
                en_model_name = get_model_name(languages[src_lang], "en")
                en_model = MarianMTModel.from_pretrained(en_model_name)
                en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)

                en_translated_text = translate_text(input_text, en_model, en_tokenizer)

                tgt_model_name = get_model_name("en", languages[tgt_lang])
                tgt_model = MarianMTModel.from_pretrained(tgt_model_name)
                tgt_tokenizer = MarianTokenizer.from_pretrained(tgt_model_name)

                translated_text = translate_text(en_translated_text, tgt_model, tgt_tokenizer)

            return render(request, 'translator/translation_result.html', {'form': form, 'translated_text': translated_text})

    else:
        form = TranslationForm()

    return render(request, 'translator/translation.html', {'form': form})
