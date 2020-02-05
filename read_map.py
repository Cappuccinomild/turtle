import cv2

build_map_name = 'built_map.png'
turtle_bot_radius = 3
wall = 0
point_color = 127
img = cv2.imread(build_map_name, 2)
print(img)
point = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j] == 127:
            point.append([i, j])
            img[i][j] = 255

print(point)

find_wall = ([1, 0], [0, 1], [-1, 0], [0, -1])
find_way = ([3, 0], [0, 3], [-3, 0], [0, -3])
moved_point = []
img = cv2.imread(build_map_name, 2)
print(img)
for x, y in point:
    print(x, y)
    if (x - 1) >= 0 and (x + 3) < img.shape[0]:
        if img[x - 1][y] == wall:
            img[x + turtle_bot_radius][y] = point_color
            moved_point.append([x + turtle_bot_radius, y])
            continue
    if (x - 3) >= 0 and (x + 1) < img.shape[0]:
        if img[x + 1][y] == wall:
            img[x - turtle_bot_radius][y] = point_color
            moved_point.append([x - turtle_bot_radius, y])
            continue
    if (y - 1) >= 0 and (y + 3) < img.shape[1]:
        if img[x][y - 1] == wall:
            img[x][y + turtle_bot_radius] = point_color
            moved_point.append([x, y + turtle_bot_radius])
            continue
    if (y - 3) >= 0 and (y + 1) < img.shape[1]:
        if img[x][y + 1] == wall:
            img[x][y - turtle_bot_radius] = point_color
            moved_point.append([x, y - turtle_bot_radius])
            continue
pointed_map_name = 'pointed_map.png'
cv2.imwrite(pointed_map_name, img)
print(moved_point)

