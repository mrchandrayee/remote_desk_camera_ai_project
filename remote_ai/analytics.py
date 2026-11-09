import cv2

class MotionDetector:
    def __init__(self, threshold=22, min_area=1200):
        self.threshold = threshold
        self.min_area = min_area
        self.background = None

    def apply(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self.background is None:
            self.background = gray.astype("float")
            return frame, False, 0
        cv2.accumulateWeighted(gray, self.background, 0.025)
        delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.background))
        thresh = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) >= self.min_area]
        for c in contours:
            if cv2.contourArea(c) >= self.min_area:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,180,255),2)
                cv2.putText(frame,"MOTION",(x,max(20,y-6)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,180,255),2)
        return frame, bool(areas), int(max(areas, default=0))
