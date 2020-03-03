import speech_recognition as sr 
from urllib.request import urlopen


if __name__=="__main__":
        
    r = sr.Recognizer()

    with sr.WavFile(urlopen('http://130.229.146.176:8080/audio.wav')) as source:
        print("Say something!")
        audio = r.listen(source, phrase_time_limit=1)

    # recognize speech using Google Speech Recognition
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        print("Google Speech Recognition thinks you said " + r.recognize_google(audio))
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
