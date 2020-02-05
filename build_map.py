import cv2

turtle_bot_radius = 3
wall = 0
road = 254
gray_zone = 195
origin_map_name = 'origin_map.png'
build_map_name = 'built_map.png'

img = cv2.imread(origin_map_name, 2)
direction = [[i, j] for i in range(- turtle_bot_radius + 1, turtle_bot_radius) for j in range(-turtle_bot_radius + 1, turtle_bot_radius)]
direction.remove([0, 0])
direction.extend([[turtle_bot_radius, 0], [-turtle_bot_radius, 0], [0, turtle_bot_radius], [0, -turtle_bot_radius]])
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j] == wall:
            for k, l in direction:
                if ((i + k) < wall or (i + k) >= img.shape[0]) or ((j + l) < wall or (j + l) >= img.shape[1]):
                    continue
                if (img[i + k][j + l] != road) and (img[i + k][j + l] != gray_zone):
                    continue
                img[i+k][j+l] = gray_zone
                
cv2.imwrite(build_map_name, img)
