from heapq import *
import cv2
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

global path_adj

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

			if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
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


def init_adj(X, img_name):
	img = cv2.imread(img_name, 2)
	path_adj = [[0 for col in range(len(X))] for row in range(len(X))]
	#make adj list
	for i in range(len(X)):
		for j in range(i, len(X)):
			path_adj[i][j] = len(astar(img, X[i], X[j]))

	for i in range(len(X)):
		for j in range(i, len(X)):
			path_adj[j][i] = path_adj[i][j]
	'''
	for i in range(len(X)):
	print(path_adj[i])
	'''
	return path_adj


# init
def init_kmean(K, X, img):

	mean = set()
	X_label = []

	path_adj = init_adj(X, 'newmap.png')

	# 초기 점 설정

	# 첫번째 점은 랜덤
	point = random.randint(0, len(X) - 1)

	# 이전에 선택된 점과의 거리를 기준으로 가장 먼 점이 높은 확률로 선정되게 한다

	while len(mean) < K:
		sample_rate = []
		#0~1까지의 확률을 가진 실수 리스트
		total = sum(path_adj[point])

		for i in range(len(X)):
			sample_rate.append(path_adj[i][point]/total)

	    #선택
		select = random.random()

		index = 0
		for i in sample_rate:
			select -= i

			if select <= 0:
				break

			index += 1

		mean.add(point)
		point = index

	mean = list(mean)
	mean.sort()

	#초기점에따라 군집화
	cnt = 0
	for x_i in range(len(X)):
		if cnt in mean:
			X_label.append(mean.index(cnt))
			cnt += 1
			continue

		s = 9999
		label = 0
		for i in mean:
			length = path_adj[x_i][i]
			#length = len(astar(img, X[x_i], X[i]))
			if s > length:
				s = length
				label = i

		X_label.append(mean.index(label))

		cnt += 1

	return X_label


# calc G
def calc_G(K, X, img, X_label):
	mean=[]
	label_cnt=[]

	for i in range(K):
		mean.append([0, 0])
		label_cnt.append(0)

	mean = np.int_(mean)
	label_cnt = np.array(label_cnt)

	for i in range(len(X)):
		mean[X_label[i]] += X[i]
		label_cnt[X_label[i]] += 1
	# print(mean)
	# print(label_cnt)
	# print()

	for i in range(len(mean)):
		mean[i] = is_wall(img, mean[i] / label_cnt[i])
	return mean


def is_wall(img, p):
	row, col = p
	height, width = img.shape
	# print(row, col)
	row = int(row)
	col = int(col)
	flag = 0
	if img[row][col] == 0:  # 중심점이 벽일경우
		while True:
			if row - flag > 0:
				if img[row - flag][col] != 0:  # 위쪽이 검은색X
					return np.array([row - flag, col])
			if row + flag < width:
				if img[row + flag][col] != 0:  # 아래쪽이 검은색X
					return np.array([row + flag, col])
			if col - flag > 0:
				if img[row][col - flag] != 0:  # 왼쪽이 검은색X
					return np.array([row, col - flag])
			if col + flag < width:
				if img[row][col + flag] != 0:  # 오른쪽이 검은색X
					return np.array([row, col+flag])
			flag = flag + 1
	else:
		return np.array([row, col])


def write_to_img(img_name, X, X_label):
	img = cv2.imread(img_name)
	color_list = []

	while len(color_list) <= max(X_label)+1:
		color = list(np.random.random(size=3) * 255)
		while color in color_list:
			color = list(np.random.random(size=3) * 255)

		color_list.append(color)
	for i in range(len(X)):
		row = X[i][0]
		col = X[i][1]
		img[row][col] = color_list[X_label[i]]
		'''
		cv2.imshow('result', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		'''
	cv2.imwrite('result.png', img)


def write_to_imgf(img_name, X):
	img = cv2.imread(img_name)
	color_list = []

	while len(color_list) <= len(X):
		color = list(np.random.random(size=3) * 255)
		while color in color_list:
			color = list(np.random.random(size=3) * 255)

		color_list.append(color)

	for x in X:
		color = color_list.pop()
		for f in x:
			row = f[0]
			col = f[1]
			img[row][col] = color

	cv2.imwrite('resultf.png', img)


def run_kmeans(K, X, img_name):
	img = cv2.imread(img_name, 2)
	# X = np.array([[[40,1], [42, 10]], [[10, 34], [21,16], [2, 23]], [[37, 26], [39, 37], [37, 41], [47, 48], [48,12]]])
	# X = np.array([[40,1], [42, 10], [10, 34], [21,16], [2, 23], [37, 26], [39, 37], [37, 41], [47, 48], [48,12]])
	X_label = init_kmean(K, X, img)
	mean = calc_G(K, X, img, X_label)
	pre1 = mean
	pre2 = mean

	# run kmean
	while True:
		temp_label = []
		density = []
		for i in range(K):
			density.append(0)
		cnt= 0
		for x in X:
			s = 9999
			label = 0
			for i in range(len(mean)):
				length = len(astar(img, x, list(mean[i])))
				if s > length:
					s = length
					label = i
			density[label] += s**2
			temp_label.append(label)
			cnt+=1

		if X_label == temp_label:
			break
		X_label = copy.deepcopy(temp_label)
		mean = calc_G(K, X, img, temp_label)
		print(pre1)
		print(pre2)
		print(mean)
		print()
		count1 = 1
		count2 = 1
		for j in range(0, K):
			for i in range(0, 2):
				if pre2[j][i] != mean[j][i]:
					count2 = 0
					break
				if pre1[j][i] != mean[j][i]:
					count1 = 0
					break
		if count1 == 1 or count2 == 1:
			return X_label, density
		pre2 = pre1
		pre1 = mean
	return X_label, density


def elbow(inertias):
	opt = []
	for i in range(1, len(inertias)-1):
		opt.append((inertias[i-1] - inertias[i]) / (inertias[i] - inertias[i+1]))

	return opt.index(max(opt)) + 3


def display_tsp(K,X_label,X):
	list_tsp=[]
	for i in range(K):
		list_tsp.append([])

	for i in range(len(X)):
		list_tsp[X_label[i]].append(X[i])
	for i in range(K):
		print(i,"번째: ",list_tsp[i],len(list_tsp[i]))

	for i in range(K):
		tsp_path_adj = init_adj(list_tsp[i], 'newmap.png')
		r = range(len(tsp_path_adj))
		# Dictionary of distance
		dist = {(i, j): tsp_path_adj[i][j] for i in r for j in r}
		path_len,path=tsp.tsp(r, dist)


if __name__ == '__main__':

	mean_dist = []

	X = np.array([[1, 6], [1, 23], [2, 2], [2, 12], [2, 16], [2, 17], [3, 6], [3, 28], [3, 31], [4, 36], [5, 15], [5, 36], [6, 5], [6, 9], [6, 19], [6, 22], [9, 3], [9, 12], [9, 16], [10, 3], [10, 36], [11, 22], [11, 27], [11, 34], [12, 27], [12, 34], [13, 3], [13, 6], [13, 7], [13, 17], [13, 20], [14, 12], [14, 25], [14, 34], [15, 31], [15, 34], [16, 15], [16, 20], [17, 7], [17, 10], [17, 22], [18, 2], [19, 10], [19, 30], [19, 33], [19, 34], [19, 38], [20, 5], [21, 7], [21, 10], [21, 14], [21, 16], [21, 17], [21, 19], [21, 20], [21, 27], [22, 10], [22, 32], [23, 2], [23, 5], [23, 7]])

	path_adj = init_adj(X, 'newmap.png')

	for i in range(2, 9):
		print(i)
		X_label, density = run_kmeans(i, X, 'newmap.png')
		mean_dist.append(sum(density))


	'''
	print(mean_dist)

	x = range(2, 10)
	y = mean_dist

	plt.plot(x, y, marker='o')
	plt.show()
	'''

	write_to_img('newmap.png', X, X_label)

	e=elbow(mean_dist)

	print("elbow : ", e)

	K = e
	X_label, density = run_kmeans(K, X, 'newmap.png')

	sep = [[] for row in range(K)]

	for i in range(len(X)):
		sep[X_label[i]].append(list(X[i]))

	write_to_img('newmap.png', X, X_label)

	print(sep)
	i = 0
	while True:
		print(i, len(sep))
		if i >= len(sep):
			break

		if len(sep[i]) > 15:

			X_label, density = run_kmeans(4, sep[i], 'newmap.png')

			temp_sep = [[] for row in range(4)]

			for j in range(len(sep[i])):
				temp_sep[X_label[j]].append(sep[i][j])

			print("before", sep)
			print()
			for t in temp_sep:
				sep.append(t)
			print("after", sep)
			print()

		i+=1

	write_to_imgf('newmap.png', sep)
