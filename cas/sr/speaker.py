# https://cloud.google.com/text-to-speech/docs/reference/libraries#client-libraries-install-python
from googletrans import Translator
from gtts import gTTS
import os


def translate(in_text, dest_language='en'):
    translator = Translator()
    # a = translator.translate('de viribus et motibus', dest='en')
    result = translator.translate(in_text, dest=dest_language)
    print('Translation: ' + result.text + '. Translated language is ' + dest_language)
    return result


translate('De viribus et motibus fluidorum commentaril')  #Example use-case

tts = gTTS(text="This is the pc speaking", lang='en')
tts.save("pcvoice.mp3")
# to start the file from python
os.system("mpg321 pcvoice.mp3")