import cv2
import sys
import math
import numpy
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

map_data = cv2.imread('./newmap.png', cv2.IMREAD_GRAYSCALE)
height, width = map_data.shape

cvt_map = map_data.tolist()

for i in range(height):
    for j in range(width):
        if cvt_map[i][j] == 0:
            cvt_map[i][j] = 0
        elif cvt_map[i][j] == 127:
            cvt_map[i][j] = 1
        else:
            cvt_map[i][j] = 1
print(cvt_map)

start = [27,14]
end = [12,52]

grid = Grid(matrix=cvt_map)

start = grid.node(6, 1)
end = grid.node(28, 3)

finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
path, runs = finder.find_path(start, end, grid)
print(path)
print('operations:', runs, 'path length:', len(path))
#print(grid.grid_str(path=path, start=start, end=end))

#print(grid.grid_)

for i, j in path:
    map_data[j][i] = int(255/3)

cv2.imwrite('path.png', map_data)
