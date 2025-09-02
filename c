import cv2
import numpy as np

# Convolution function
def apply_mask(image, mask):
    m, n = mask.shape
    pad = m // 2
    padded = np.pad(image, pad, mode='constant')
    output = np.zeros_like(image, dtype=np.float32)
    
    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            region = padded[i:i+m, j:j+n]
            output[i, j] = np.sum(region * mask)
    return output

# Sobel Gradient
def sobel(image):
    Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Gy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    gx = apply_mask(image, Gx)
    gy = apply_mask(image, Gy)

    # Gradient magnitude
    mag = np.sqrt(gx*2 + gy*2)
    mag = np.uint8(np.clip(mag, 0, 255))

    # Gradient angle in DEGREES
    angle = np.degrees(np.arctan2(gy, gx))
    angle[angle < 0] += 180   # Normalize to [0, 180)

    return mag, angle

# Non-Maximum Suppression
def non_maximum_suppression(mag, angle):
    rows, cols = mag.shape
    Z = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            q = 255
            r = 255
            angle_deg = angle[i, j]

            # 0°
            if (0 <= angle_deg < 22.5) or (157.5 <= angle_deg <= 180):
                q = mag[i, j+1]
                r = mag[i, j-1]
            # 45°
            elif (22.5 <= angle_deg < 67.5):
                q = mag[i+1, j-1]
                r = mag[i-1, j+1]
            # 90°
            elif (67.5 <= angle_deg < 112.5):
                q = mag[i+1, j]
                r = mag[i-1, j]
            # 135°
            elif (112.5 <= angle_deg < 157.5):
                q = mag[i-1, j-1]
                r = mag[i+1, j+1]

            if (mag[i, j] >= q) and (mag[i, j] >= r):
                Z[i, j] = mag[i, j]
            else:
                Z[i, j] = 0
    return Z

# Double Thresholding
def threshold(img, low, high):
    strong = 255
    weak = 50
    res = np.zeros_like(img, dtype=np.uint8)

    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img <= high) & (img >= low))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res, weak, strong

# Hysteresis
def hysteresis(img, weak=50, strong=255):
    rows, cols = img.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if img[i, j] == weak:
                if np.any(img[i-1:i+2, j-1:j+2] == strong):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img

# Complete Canny
def canny_edge(image, low=50, high=100):
    mag, angle = sobel(image)
    nms = non_maximum_suppression(mag, angle)
    thresh, weak, strong = threshold(nms, low, high)
    final = hysteresis(thresh, weak, strong)
    return final

# Example run
if _name_ == "_main_":
    img = cv2.imread("test.png", 0)
    edges = canny_edge(img, 50, 100)
    cv2.imwrite("canny_output.png", edges)
