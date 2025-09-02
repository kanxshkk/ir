import cv2
import matplotlib.pyplot as plt
import numpy as np
from google.colab.patches import cv2_imshow

img = cv2.imread("/content/test.jpg",cv2.IMREAD_GRAYSCALE)
cv2_imshow(img)
row,col = img.shape
scale = 2
newh, neww = row*scale, col*scale
zoh = np.zeros( (newh,neww))
for i in range(newh):
  for j in range(neww):
    zoh[i,j]= img[i//2,j//2]
cv2_imshow(zoh)
newh, neww = row*scale, col*scale
foh = np.zeros((newh,neww))
gray_value = [i for i in range(0,256)]
freq = [0]*256
img = cv2.imread("/content/test.jpg",cv2.IMREAD_GRAYSCALE)
r,c = img.shape

gray_value = [i for i in range(0,256)]
freq = [0]*255

for i in range(r):
  for j in range(c):
    freq[img[i,j]] +=1

prob = [f/sum(freq) for f in freq]


cdf = np.cumsum(prob) * (255)
sk = np.round(cdf,0)
new_p = [0]*256
for index,intensity in enumerate(list(sk)):
  new_p[int(intensity)]+=prob[index]
  plt.bar(gray_value, new_p)
plt.title("Histogram of Equalized Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()
plt.bar(gray_value, prob)
plt.title("Histogram of Equalized Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()
new_img = np.zeros(img.shape)
for i in range(img.shape[0]):
  for j in range(img.shape[1]):
    new_img[i,j] = sk[img[i,j]]
cv2_imshow(new_img)
cv2_imshow(img)
