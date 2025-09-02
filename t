def otsu_threshold(img):
    # Histogram
    hist = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i,j]] += 1

    total = img.size
    sum_total = np.dot(np.arange(256), hist)  # total weighted sum
    sumB, wB, max_var, threshold = 0.0, 0.0, 0.0, 0

    for t in range(256):
        wB += hist[t]       # weight background
        if wB == 0: continue
        wF = total - wB     # weight foreground
        if wF == 0: break
        sumB += t * hist[t]
        mB = sumB / wB      # mean background
        mF = (sum_total - sumB) / wF  # mean foreground
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t

    return binary_threshold(img, threshold), threshold

otsu_img, otsu_T = otsu_threshold(img)
cv2.imwrite("otsu_thresh.png", otsu_img)
print("Otsu Threshold =", otsu_T)
