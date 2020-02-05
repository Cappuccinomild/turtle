import cv2
import numpy as np

img = cv2.imread('map.pgm', 2)

#위 아래 자르기
result = []
add= False
for row in img:
    for i in row:
        if i < 200:
            add = True

    if add:
        result.append(row)

    add = False

#rotate
img = np.array(result)
result = []
for j in range(len(img[0])):
    temp = []
    for i in range(len(img)):
        temp.append(img[i][j])

    result.append(temp)

#위 아래 자르기
img = np.array(result)

result = []
add= False
for row in img:
    for i in row:
        if i < 200:
            add = True


    if add:
        result.append(row)

    add = False

#rotate
img = np.array(result)
result = []
for j in range(len(img[0])):
    temp = []
    for i in range(len(img)):
        temp.append(img[i][j])

    result.append(temp)

result = np.array(result)
cv2.imwrite('map.png', result)
