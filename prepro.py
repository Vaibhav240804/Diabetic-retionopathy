import cv2
import numpy as np
import matplotlib
image_size = [224,224]
def preprocess_image(img):
    # if img.shape[-1] == 1:
    #   img = cv2.merge([img]*3)
    # Read the image
    # Remove black borders
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
      cnt = contours[0]
      x, y, w, h = cv2.boundingRect(cnt)
      img = img[y:y+h, x:x+w]
    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Adjust brightness and contrast
    alpha = 1.5 # Contrast control (1.0-3.0)
    beta = 50 # Brightness control (0-100)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    # Convert to grayscale
    
    
    # Resize to 64x64 pixels
    img = cv2.resize(img, image_size)
    
    # Convert to float32 and normalize
    img = img.astype('float32') / 255.0
    # Add a channel dimension and return the preprocessed image
    img = np.expand_dims(img, axis=0)
    cv2.imshow(img)
    return img

img = cv2.imread('./static/uploads/0ac436400db4.png')
