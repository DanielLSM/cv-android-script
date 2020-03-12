# https://cloud.google.com/text-to-speech/docs/reference/libraries#client-libraries-install-python
from googletrans import Translator
from gtts import gTTS
import os


class Speaker:
    def __init__(self, dest_language='es'):
        self.dest_language = dest_language
        self.translator = Translator()
        self.starter_sentence = "This object is a "

    def speak(self, in_text):
        assert isinstance(in_text, str)
        tts = gTTS(text=self.starter_sentence + in_text, lang='en')
        tts.save("speak.mp3")
        os.system("mpg321 speak.mp3")

    def speak_translation(self, in_text):
        assert isinstance(in_text, str)
        tts = gTTS(text=self.starter_sentence + in_text, lang=self.dest_language)
        tts.save("speak_translate.mp3")
        os.system("mpg321 speak_translate.mp3")

    def dialog(self, in_text):
        self.speak(in_text)
        self.speak_translation(in_text)


def translate(in_text, dest_language='en'):
    translator = Translator()
    result = translator.translate(in_text, dest=dest_language)
    print('Translation: ' + result.text + '. Translated language is ' + dest_language)
    return result


if __name__ == '__main__':

    speaker = Speaker()
    speaker.dialog("hello")