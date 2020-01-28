import cv2
from sklearn.cluster import KMeans
import numpy as np
img = cv2.imread('map.png', 2)
print(img)
X = np.array([[40,1], [42, 10], [10, 34], [21,16], [2, 23], [37, 26], [39, 37], [37, 41], [47, 48], [48,12]])
kmeans = KMeans(n_clusters=2, random_state=0, img = img).fit(X)
print(kmeans.labels_)
cnt = 0
for i, j in X:
	img[i][j] = kmeans.labels_[cnt] * 100 + 100
	cnt += 1

cv2.imwrite('result.png', img)
