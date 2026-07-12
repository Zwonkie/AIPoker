import cv2

tpl = cv2.imread("card_templates/dealer_button.png")
print("Template dimensions:", tpl.shape if tpl is not None else "None")

# In hero_crop.png, the dealer button is on the top-left.
# Let's crop it from hero_crop.png to see its size.
# In hero_crop.png, the button seems to be at y: 65 to 115, x: 0 to 45
crop_img = cv2.imread("scratch/hero_crop.png")
if crop_img is not None:
    button_crop = crop_img[65:120, 0:50]
    cv2.imwrite("scratch/actual_button.png", button_crop)
    print("Actual button crop dimensions:", button_crop.shape)
