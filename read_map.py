import cv2

img = cv2.imread('map2.png', 2)
print(img.shape)
print(img)
point = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j] == 127:
            point.append([i,j])

print(point)
