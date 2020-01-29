# Real-Time-Morse-Video-to-Audio-Decoder
Formulated an audio decoder which takes input a sequence of dots and dashes (in form of gestures) and outputs the corresponding audio using the morse code. 

# How-To_Run
Run MorseDecoder.py to run the project.  
Read the file DSPLabFinalReport.pdf for project report.  

In line 156  
_, contours= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
if the code gives error change to:  
_, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

The libraries needed are:  
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
import os  

# Project-In-Action  

Youtube Link: https://youtu.be/GmsajcI1HdA  


