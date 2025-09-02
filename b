div_img = np.clip(A.astype(float) / (B.astype(float)+1), 0, 255).astype(np.uint8)
cv2.imwrite("div_output.png", div_img)

and_img[i, j] = a & b
