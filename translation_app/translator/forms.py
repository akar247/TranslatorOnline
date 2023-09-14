from django import forms

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


class TranslationForm(forms.Form):
    original_language = forms.ChoiceField(choices=[(lang, lang.capitalize()) for lang in languages.keys()])
    target_language = forms.ChoiceField(choices=[(lang, lang.capitalize()) for lang in languages.keys()])
    input_text = forms.CharField(widget=forms.Textarea)


from django.shortcuts import render
from .forms import TranslationForm
from .translator import translate_text

def translation_view(request):
    if request.method == 'POST':
        form = TranslationForm(request.POST)
        if form.is_valid():
            src_lang = form.cleaned_data['original_language'].lower()
            tgt_lang = form.cleaned_data['target_language'].lower()
            input_text = form.cleaned_data['input_text']

            model_name = get_model_name(languages[src_lang], languages[tgt_lang])
            model = MarianMTModel.from_pretrained(model_name)
            tokenizer = MarianTokenizer.from_pretrained(model_name)

            translated_text = translate_text(input_text, model, tokenizer)
            return render(request, 'translator/translation_result.html', {'form': form, 'translated_text': translated_text})

    else:
        form = TranslationForm()

    return render(request, 'translator/translation.html', {'form': form})

