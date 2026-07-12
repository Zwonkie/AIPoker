import cv2
import pytesseract
import sys
import os

# Add project root to sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vision import PokerVision

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread("scratch/pot_crop_91656.png")
vision = PokerVision()

# OCR with whitelist
text = vision.ocr_roi(img, (0, 0, img.shape[1], img.shape[0]), whitelist='0123456789.Pulje: ')
print("OCR raw output with whitelist:", repr(text))

# OCR without whitelist
gray = vision.preprocess_image(img)
resized = cv2.resize(gray, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
raw_no_whitelist = pytesseract.image_to_string(resized, config='--psm 6').strip()
print("OCR raw output without whitelist:", repr(raw_no_whitelist))

val = vision.clean_pot_string(text)
print("Cleaned pot value:", val)
