import cv2

hero_crop = cv2.imread("scratch/hero_crop.png")
if hero_crop is not None:
    # Let's crop the button.
    # We want to crop just the yellow button with a bit of green margin around it.
    # The yellow button center is roughly around y=65 in hero_crop.
    button = hero_crop[40:90, 0:45]
    cv2.imwrite("scratch/perfect_dealer.png", button)
    print("Saved scratch/perfect_dealer.png with shape:", button.shape)
else:
    print("hero_crop not found")
