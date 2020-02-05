import cv2

turtlebot_radius=3
img = cv2.imread('origin_map.png', 2)
direction=[[i,j] for i in range(-2,3) for j in range(-2,3)]
direction.remove([0,0])
direction.extend([[3,0],[-3,0],[0,3],[0,-3]])
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j]==0: #현재가 검정색일 때
            for k,l in direction: #상하좌우 빈곳 탐지
                if ((i+k)<0 or (i+k)>=img.shape[0]) or ((j+l)<0 or (j+l)>=img.shape[1]):
                    continue
                if (img[i+k][j+l] != 255) and (img[i+k][j+l] != 120):
                    continue
                img[i+k][j+l]=120
cv2.imwrite('build_map2.png', img)
