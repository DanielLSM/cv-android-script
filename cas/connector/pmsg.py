import copy, cv2, numpy, time


#Install cv2 and numpy to run. pip install opencv-python and pip install opencv-python-headless
class FrameCapturer:
    def __init__(self, camera_id, ratio=0.5):
        self.frame = None
        self.ratio = ratio
        self.open_camera(camera_id)

# This opens the frame by by camera_id. Default camera is zero. If you attach more camera you need to specify

    def open_camera(self, camera_id):
        self.cap = cv2.VideoCapture(camera_id)
        print(self.cap)
        self.fw = self.cap.get(3)
        self.fl = self.cap.get(4)
        self.rw = int(self.cap.get(3) * self.ratio)
        self.rl = int(self.cap.get(4) * self.ratio)
# Fixes the frame to designed rectangle. Default is 256-256

    def set_frame(self):
        cv2.rectangle(self.frame, (int((self.fw - self.rw) / 2), int(
            (self.fl - self.rl) / 2)), (int((self.fw + self.rw) / 2), int((self.fl + self.rl) / 2)),
                      (0, 250, 460), 3)
# Returns the center of the frame as a x,y tuple

    def set_center(self):
        cv2.circle(self.frame, (int(self.cap.get(3) / 2), int(self.cap.get(4) / 2)), 4,
                   (0, 250, 460), 4, 5)


# Send the deep copy not the actualy frame. This can be modified safely

    def get_frame(self):
        # self.open_camera(0)
        self.ret, self.frame = self.cap.read()
        subframe = self.get_subframe()
        self.set_frame()
        self.set_center()
        return self.frame, subframe

    def get_subframe(self):
        # cv2.selectROI(self.frame)
        return copy.copy(self.frame[int((self.fl - self.rl) / 2):int((self.fl + self.rl) / 2),
                                    int((self.fw - self.rw) / 2):int((self.fw + self.rw) / 2)])

    def clear(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    #EXAMPLE

    url = "http://130.229.170.85:8080/video"
    a = FrameCapturer(url)
    # a = FrameCapturer(0)

    while True:
        frame, subframe = a.get_frame()
        cv2.imshow('sub_frame', subframe)
        # cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            a.clear()
            break
