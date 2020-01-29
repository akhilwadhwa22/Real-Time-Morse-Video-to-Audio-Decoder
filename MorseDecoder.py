import cv2
import math
import numpy as np
import pyautogui
from statistics import mode
import wave
import pyaudio
from pygame import mixer
from gtts import gTTS 
import tkinter as Tk


xs = [6.0/20.0, 9.0/20.0, 12.0/20.0]
ys = [9.0/20.0, 10.0/20.0, 11.0/20.0]
pyautogui.PAUSE = 0
bgSubtractor = False
isBgCaptured = None
def createHandHistogram(frame):
        rows, cols, _ = frame.shape
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi = np.zeros([180, 20, 3], dtype=hsvFrame.dtype)

        i = 0
        for x in xs:
            for y in ys:
                x0, y0 = int(x*rows), int(y*cols)
                roi[i*20:i*20 + 20, :, :] = hsvFrame[x0:x0 + 20, y0:y0 + 20, :]

                i += 1
        handHist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        return cv2.normalize(handHist, handHist, 0, 255, cv2.NORM_MINMAX)
    
def drawRect(frame):
        rows, cols, _ = frame.shape

        for x in xs:
            for y in ys:
                x0, y0 = int(x*rows), int(y*cols)
                cv2.rectangle(frame, (y0, x0), (y0 + 20, x0 + 20), (0, 255, 0), 1)
                
def threshold(mask):
        grayMask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(grayMask, 0, 255, 0)
        return thresh
    
def getextreme (cnt):
        c = max(cnt, key=cv2.contourArea)
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        #print(extTop)
        
        return extTop, c

def getMaxContours(contours):
        maxIndex = 0
        maxArea = 0

        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)

            if area > maxArea:
                maxArea = area
                maxIndex = i
        return contours[maxIndex]

def setupFrame(frame_width, frame_height):
        """self.x0 and self.y0 are top left corner coordinates
        self.width and self.height are the width and height the ROI
        """
        x, y = 0.1, 0.05
        x0 = int(frame_width*x)
        y0 = int(frame_height*y)
        width = 280
        height = 280
        
        return x0, y0, height, width
        
def getCentroid(contour):
        moment = cv2.moments(contour)
        if moment['m00'] != 0:
            cx = int(moment['m10']/moment['m00'])
            cy = int(moment['m01']/moment['m00'])
            return [cx, cy]
        else:
            return None
        
def calculateAngle(far, start, end):
        """Cosine rule"""
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = math.acos((b**2 + c**2 - a**2) / (2*b*c))
        return angle
        
def countFingers(contour, contourAndHull):
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(contour, hull)
            cnt = 0
            if type(defects) != type(None):
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s, 0])
                    end = tuple(contour[e, 0])
                    far = tuple(contour[f, 0])
                    angle = calculateAngle(far, start, end)

                    # Ignore the defects which are small and wide
                    # Probably not fingers
                    if d > 10000 and angle <= math.pi/2:
                        cnt += 1
                        cv2.circle(contourAndHull, far, 8, [255, 0, 0], -1)
            return True, cnt
        return False, 0
        
def histMasking(frame, handHist):
        """Create the HSV masking"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], handHist, [0, 180, 0, 256], 1)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        cv2.filter2D(dst, -1, disc, dst)

        ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=7)
        thresh = cv2.merge((thresh, thresh, thresh))
        return cv2.bitwise_and(frame, thresh)
def detectHand(frame, handHist, bgSubtractor, x0, y0, height, width):
        bgSubtractorLr = 0
        roi = frame[y0:y0 + height, 
                x0:x0 + width,:]
        dist = 0
        cnt = 0
        roi = cv2.bilateralFilter(roi, 5, 50, 100)
        # Color masking
        histMask = histMasking(roi, handHist)
        cv2.imshow("histMask", histMask)

        # Background substraction
        fgmask = bgSubtractor.apply(roi, learningRate=bgSubtractorLr)
        kernel = np.ones((4, 4), np.uint8)
        # MORPH_OPEN removes noise
        # MORPH_CLOSE closes the holes in the object
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
        bgSubMask = cv2.bitwise_and(roi, roi, mask=fgmask)
        cv2.imshow("bgSubMask", bgSubMask)
        # Overall mask
        mask = cv2.bitwise_and(histMask, bgSubMask)

        thresh = threshold(mask)
        cv2.imshow("Overall thresh", thresh)

        _,contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        if len(contours) > 0:
            maxContour = getMaxContours(contours)
            # Draw contour and hull
            contourAndHull = np.zeros(roi.shape, np.uint8)
            hull = cv2.convexHull(maxContour)
            cv2.drawContours(contourAndHull, [maxContour], 0, (0, 255, 0), 2)
            cv2.drawContours(contourAndHull, [hull], 0, (0, 0, 255), 3)
            extTop, c = getextreme(contours)
            cv2.drawContours(contourAndHull, [c], -1, (0, 255, 0), 2)
            cv2.circle(contourAndHull, extTop, 8, (255, 0, 0), -1)
            found, cnt = countFingers(maxContour, contourAndHull)
            
            if found:
                if cnt == 1:
                    pyautogui.press("down")
                elif cnt == 2:
                    pyautogui.press("up")

            centroid = getCentroid(maxContour)
            if centroid is not None:
                cv2.circle(contourAndHull, tuple(centroid), 8, [102, 255, 255], 2)
                cv2.line(contourAndHull,tuple(centroid),extTop,[102, 255, 255],2)
            dist = math.sqrt((extTop[0] - centroid[0])**2 + (extTop[1] - centroid[1])**2)

            cv2.imshow("Contour and Hull", contourAndHull)
        return cnt, dist, len(contours)   
    
def MorseDetector(gestures):
    s=""
    for g in gestures[:-1]:
        if g==1 or g==2:
            s=s+"1"
        else:
            s=s+str(g)
            
    dictionary = {
    "01": {"letter": "A", "sound": "a.wav"},
    "1000": {"letter": "B", "sound": "b.wav"},
    "1010": {"letter": "C", "sound": "c.wav"},
    "100": {"letter": "D", "sound": "d.wav"},
    "0": {"letter": "E", "sound": "e.wav"},
    "0010": {"letter": "F", "sound": "f.wav"},
    "110": {"letter": "G", "sound": "g.wav"},
    "0000": {"letter": "H", "sound": "h.wav"},
    "00": {"letter": "I", "sound": "i.wav"},
    "0111": {"letter": "J", "sound": "j.wav"},
    "101": {"letter": "K", "sound": "k.wav"},
    "0100": {"letter": "L", "sound": "l.wav"},
    "11": {"letter": "M", "sound": "m.wav"},
    "10": {"letter": "N", "sound": "n.wav"},
    "111": {"letter": "O", "sound": "o.wav"},
    "0110": {"letter": "P", "sound": "p.wav"},
    "1101": {"letter": "Q", "sound": "q.wav"},
    "010": {"letter": "R", "sound": "r.wav"},
    "000": {"letter": "S", "sound": "s.wav"},
    "1": {"letter": "T", "sound": "t.wav"},
    "001": {"letter": "U", "sound": "u.wav"},
    "0001": {"letter": "V", "sound": "v.wav"},
    "011": {"letter": "W", "sound": "w.wav"},
    "1001": {"letter": "X", "sound": "x.wav"},
    "1011": {"letter": "Y", "sound": "y.wav"},
    "1100": {"letter": "Z", "sound": "z.wav"}   
    }
    try:
        letter = dictionary[s]["letter"]
        file = dictionary[s]["sound"]
        valid  = True
    except:
        letter = "none"
        valid  = False

    if valid:
        
        mixer.init()
        mixer.music.load(file)
        mixer.music.play()
        
    
    return(letter, valid)
    
def calibrate_hand():
    global isHandHistCreated
    isHandHistCreated = True
    global handHist
    handHist = createHandHistogram(frame)   
    
def start_detection():
    global bgSubtractor
    bgSubtractor = cv2.createBackgroundSubtractorMOG2(10, bgSubThreshold)
    global isBgCaptured
    isBgCaptured = True
    
def reset_capture():
    global bgSubtractor
    bgSubtractor = None
    global isBgCaptured
    isBgCaptured = False

def quit_prog():
    cap.release()
    root.destroy()
    cv2.destroyAllWindows()
    
def turnDown():
    global y0,frame_height,height
    y0 = min(y0 + 20, frame_height - height)
def turnUp():
    global y0
    y0 = max(y0 - 20, 0) 
def turnLeft(): 
    global x0
    x0 = max(x0 - 20, 0)
def turnRight():
    global x0,frame_width,width
    x0 = min(x0 + 20, frame_width - width)    
    
    
if __name__ == "__main__":
    root = Tk.Tk()
    root.title('Output Text')
    root.geometry("700x700")
    s1 = Tk.StringVar()
    s2 = Tk.StringVar()
    L1 = Tk.Label(root, textvariable = s1)
    L2 = Tk.Label(root, textvariable = s2)
    L1.pack(side=Tk.TOP)
    L2.pack(side=Tk.TOP)
    L1.config(width=25, height=3, font=("Helvetica",30))
    L2.config(width=25, height=3, font=("Helvetica",30))
    B1 = Tk.Button(root, text = 'Calibrate hand', command = calibrate_hand,font=("Helvetica",15))
    B2 = Tk.Button(root, text = 'Start detection', command = start_detection,font=("Helvetica",15))
    B3 = Tk.Button(root, text = 'Reset', command = reset_capture,font=("Helvetica",15))
    B4 = Tk.Button(root, text = 'Quit', command = quit_prog,font=("Helvetica",15))
    B1.pack(padx=5, pady=10, side=Tk.LEFT)
    B2.pack(padx=5, pady=10, side=Tk.LEFT)
    B3.pack(padx=5, pady=10, side=Tk.RIGHT)
    B4.pack(padx=5, pady=10, side=Tk.RIGHT)
    
    B5 = Tk.Button(root, text = 'Down', command = turnDown,font=("Helvetica",15))
    B6 = Tk.Button(root, text = 'Up', command = turnUp,font=("Helvetica",15))
    B7 = Tk.Button(root, text = 'Right', command = turnRight,font=("Helvetica",15))
    B8 = Tk.Button(root, text = 'Left', command = turnLeft,font=("Helvetica",15))
    B5.pack(padx=5, pady=10, side=Tk.BOTTOM)
    B7.pack(padx=5, pady=10, side=Tk.BOTTOM)
    B8.pack(padx=5, pady=10, side=Tk.BOTTOM)
    B6.pack(padx=5, pady=10, side=Tk.BOTTOM)
    
    
    fingers = 0
    sentence = ''
    gestures = []
    gestures_list = []
    word = ''
    max_gestures = 0
    word_flag= 0
    append_flag = 0
    hand_flag = 0
    play_flag = 0
    isHandHistCreated = False
    isBgCaptured = False
    bgSubThreshold = 30
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,60)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,20)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.3)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*0.3)
    x0, y0, height, width = setupFrame(frame_width, frame_height)
    
    while cap.isOpened():
        destroy=False
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (x0, y0), (x0 + width - 1, y0 + height - 1), (255, 0, 0), 2)
        k = cv2.waitKey(1) & 0xFF
        
        if isHandHistCreated and isBgCaptured:
            cnt ,dist, hand_flag= detectHand(frame, handHist, bgSubtractor, x0, y0, height, width)
            if cnt==1:
                fingers=2
            elif cnt==2:
                fingers=3
            elif cnt==3:
                fingers=4
            elif cnt==4:
            	fingers=5
            elif cnt == 0:
                if dist > 100:
                	fingers=1
                else :
                	fingers = 0   
            cv2.putText(frame, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)    
        elif not isHandHistCreated:
            drawRect(frame)
            
        if hand_flag!=0 and append_flag<31:
            gestures_list.append(fingers)
            append_flag=append_flag+1
            
        elif (hand_flag!=0 and append_flag==31):
            play_flag = 0
            try:
                max_gestures = mode(gestures_list)
                gestures.append(max_gestures)
                append_flag=append_flag+1
                s2.set("Updating character")
            except:
                append_flag=0
                gestures_list = []
                s2.set("Try again!!")
            
        elif hand_flag==0:
            append_flag=0
            gestures_list = []
            s2.set("Character Please!")
            
        if (max_gestures==5 or max_gestures==4) and play_flag==0 and word_flag==0:
            letter, valid = MorseDetector(gestures)
            if valid:
                
                word = word + letter
                sentence = sentence + letter
                s1.set(sentence)
                root.update()
                word_flag = word_flag + 1
            else:
                s2.set("Invalid Character!")
                
            gestures = []
            play_flag = 1
            
        if max_gestures!=5 and max_gestures!=4 and word_flag!=0:
            word_flag = 0
        if (max_gestures==5 or max_gestures==4) and word_flag==1 and play_flag==0:
            text = word
            speech = gTTS(text = text, lang = 'en-in', slow = False)
            speech.save('text.mp3')
            mixer.init()
            mixer.music.load('text.mp3')
            mixer.music.play()
            word_flag = 0
            play_flag = 1
            gestures = []
            sentence = sentence + ' '
            word = ''
            
        print(" Gestures: ")
        print(gestures)
             
        cv2.imshow("Output", frame)

        root.update()

    root.mainloop()
    
    cv2.destroyAllWindows()
