import cv2
import vlc
import requests
import numpy as np

if __name__ == "__main__":

    http_android = "http://130.229.146.176:8080/"
    p = vlc.MediaPlayer(http_android + "audio.wav")
    p.play()
    while True:
        url = http_android + "shot.jpg"
        img_shot = requests.get(url)
        img_array = np.array(bytearray(img_shot.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)

        cv2.imshow("AndroidCam", img)

        if cv2.waitKey(1) == 27:
            break
