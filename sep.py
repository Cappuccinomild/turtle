from heapq import *
import cv2
import numpy as np
import random
import copy
 
def heuristic(a, b):
	return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

def astar(array, start, goal):
	
	start = (int(start[0]), int(start[1]))
	goal = (int(goal[0]), int(goal[1]))
	neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
	close_set = set()
	came_from = {}
	gscore = {start:0}
	fscore = {start:heuristic(start, goal)}
	oheap = []
	heappush(oheap, (fscore[start], start))
	
	while oheap:

		current = heappop(oheap)[1]

		if current == goal:
			data = []
			while current in came_from:
				data.append(current)
				current = came_from[current]
			return data

		close_set.add(current)
		for i, j in neighbors:
			neighbor = current[0] + i, current[1] + j            
			tentative_g_score = gscore[current] + heuristic(current, neighbor)
			if 0 <= neighbor[0] < array.shape[0]:
				if 0 <= neighbor[1] < array.shape[1]:
					if array[neighbor[0]][neighbor[1]] == 0:
						continue
				else:
                    # array bound y walls
					continue
			else:
                # array bound x walls
				continue
                
			if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
				continue
                
			if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
				came_from[neighbor] = current
				gscore[neighbor] = tentative_g_score
				fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
				heappush(oheap, (fscore[neighbor], neighbor))
				
	return list(range(0,9999))
    
def calc_path(img, X, Y):
	path = []
	for x in X:
		for y in Y:
			data = astar(img, tuple(x), tuple(y))
			
			path.append(len(data))
	
	
	return np.array(path)

#init
def init_kmean(K, X, img):
        X_label = []
        mean=set()
        while len(mean) < K:
                mean.add(random.randint(0, len(X) - 1))

        mean = list(mean)
        mean.sort()

        cnt = 0
        for x in X:
                if cnt in mean:
                        X_label.append(mean.index(cnt))
                        cnt += 1
                        continue

                s = 9999
                label = 0
                for i in mean:                
                        length = len(astar(img, x, X[i]))
                        print(cnt, s, length)
                        if s > length:
                                s = length
                                label = i
                                
                X_label.append(mean.index(label))
                
                cnt += 1

        return X_label

#calc G
def calc_G(K, X, X_label):
        mean=[]
        label_cnt=[]

        for i in range(K):
                mean.append([0,0])
                label_cnt.append(0)

        mean = np.int_(mean)
        label_cnt = np.array(label_cnt)

        for i in range(len(X)):
                mean[X_label[i]] += X[i]
                label_cnt[X_label[i]] += 1

        for i in range(len(mean)):
                mean[i] = mean[i] / label_cnt[i]

        return mean


def is_wall(img, p):

        row, col = p

        if img[row-1][col] == 255 and img[row+1][col] == 255: #세로줄
                

        elif img[row][col-1] == 255 and img[row][col+1] == 255: #가로
                
        

K = 3

img = cv2.imread('map.png', 2)

#X = np.array([[[40,1], [42, 10]], [[10, 34], [21,16], [2, 23]], [[37, 26], [39, 37], [37, 41], [47, 48], [48,12]]])
X = np.array([[40,1], [42, 10], [10, 34], [21,16], [2, 23], [37, 26], [39, 37], [37, 41], [47, 48], [48,12]])

X_label = init_kmean(K, X, img)
mean = calc_G(K, X, X_label)

print(mean)
print(X_label)

#run kmean
while True:
        temp_label = []
        cnt= 0
        for x in X:

                s = 9999
                label = 0
                for i in range(len(mean)):                
                        length = len(astar(img, x, list(mean[i])))
                        print(cnt, s, length)
                        if s > length:
                                s = length
                                label = i
                                
                temp_label.append(label)
                cnt+=1
                
        print(temp_label)
        if X_label == temp_label:
                break
        X_label = copy.deepcopy(temp_label)
        mean = calc_G(K, X, temp_label)
        print(X_label)
        print(mean)

cnt = 0
for i, j in X:        
        img[i][j] = int((255/(X_label[cnt]+1))+10)
        cnt+=1
        
cv2.imwrite('result.png', img)
