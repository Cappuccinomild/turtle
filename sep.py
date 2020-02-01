from heapq import *
import cv2
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import multiprocessing

global path_adj


def astar(array, start, goal):

	start = list(start)
	goal = list(goal)

	grid = Grid(matrix=array)

	finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
	path, runs = finder.find_path(grid.node(start[1], start[0]), grid.node(goal[1], goal[0]), grid)

	return path

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

	return path_adj

def adj_row(img, X, row):

	path_adj = [0 for col in range(len(X))]

	#make adj list
	for i in range(len(X)):
		path_adj[i] = len(astar(img, X[i], X[row]))


	return path_adj



# init
def init_kmean(K, X, img):

	mean = set()
	X_label = []

	# 초기 점 설정

	# 첫번째 점은 랜덤
	point = random.randint(0, len(X) - 1)

	path_adj = adj_row(img, X, point)

	# 이전에 선택된 점과의 거리를 기준으로 가장 먼 점이 높은 확률로 선정되게 한다
	while len(mean) < K:
		sample_rate = []
		#0~1까지의 확률을 가진 실수 리스트
		total = sum(path_adj)

		for i in range(len(X)):
			sample_rate.append(path_adj[i]/total)

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
		path_adj = adj_row(img,X,point)

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
			#length = path_adj[x_i][i]
			length = len(astar(img, X[x_i], X[i]))
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


def write_to_imgf(img_name, X, out_name):
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

	cv2.imwrite(out_name, img)


def run_kmeans(args):

	K = args[0]
	X = args[1]
	img_name = args[2]

	img = cv2.imread(img_name, 2)
	#X = np.array([[[40,1], [42, 10]], [[10, 34], [21,16], [2, 23]], [[37, 26], [39, 37], [37, 41], [47, 48], [48,12]]])
    #X = np.array([[40,1], [42, 10], [10, 34], [21,16], [2, 23], [37, 26], [39, 37], [37, 41], [47, 48], [48,12]])
	X_label = init_kmean(K, X, img)
	mean = calc_G(K, X, img, X_label)
	prev = []
	# print("check")
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

		if mean in prev:
			break

		X_label = copy.deepcopy(temp_label)
		mean = calc_G(K, X, img, temp_label)
		prev.append(mean)



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
		tsp_path_adj = init_adj(list_tsp[i], 'big_map.png')
		r = range(len(tsp_path_adj))
		# Dictionary of distance
		dist = {(i, j): tsp_path_adj[i][j] for i in r for j in r}
		path_len,path=tsp.tsp(r, dist)

def get_opt_kmean(img_name, X, end):

	#인수 만들기
	args = []
	for i in range(2, end):
		args.append((i, X, img_name))

	#멀티프로세싱
	print("processing start")
	pool = multiprocessing.Pool(processes=len(args)) # 현재 시스템에서 사용 할 프로세스 개수
	result = pool.map(run_kmeans, args)
	pool.close()
	pool.join()

	#분산도 구하기
	mean_dist = []
	for density in result	:
		mean_dist.append(sum(density[1]))

	e=elbow(mean_dist)

	print("elbow : ", e)

	X_label, density = result[e - 2]#0번째의 K는 2이므로 elbow에서 2를 빼서 K를 찾음

	sep = [[] for row in range(e)]

	for i in range(len(X)):
		sep[X_label[i]].append(list(X[i]))

	return sep


if __name__ == '__main__':

	X = np.array([[4, 5], [4, 6], [4, 7], [4, 23], [4, 24], [4, 25], [4, 49], [4, 65], [4, 66], [4, 83], [5, 136], [5, 147], [6, 95], [6, 136], [6, 147], [7, 36], [7, 147], [8, 147], [11, 48], [11, 71], [12, 20], [12, 113], [12, 119], [12, 127], [14, 27], [18, 27], [19, 27], [20, 11], [20, 27], [20, 62], [22, 20], [26, 20], [27, 39], [27, 49], [27, 50], [27, 51], [27, 52], [27, 53], [27, 54], [27, 55], [27, 56], [27, 57], [27, 58], [27, 59], [27, 60], [27, 61], [27, 62], [27, 63], [27, 79], [27, 89], [30, 4], [30, 11], [31, 4], [31, 11], [34, 100], [35, 100], [36, 15], [36, 33], [36, 34], [36, 35], [36, 36], [36, 49], [36, 65], [36, 100], [39, 114], [39, 117], [40, 147], [42, 120], [43, 13], [43, 23], [43, 27], [43, 54], [43, 56], [43, 75], [43, 76], [44, 91], [44, 136], [45, 111], [49, 80], [49, 91], [51, 51], [52, 12], [52, 30], [52, 31], [52, 32], [53, 71], [53, 80], [53, 91], [55, 60], [55, 111], [56, 71], [56, 100], [56, 120], [57, 4], [57, 120], [57, 136], [58, 4], [58, 51], [58, 91], [58, 120], [59, 91], [62, 127], [63, 15], [63, 16], [63, 17], [63, 18], [63, 38], [63, 127], [63, 136], [64, 60], [64, 152], [64, 153], [64, 154], [65, 80], [66, 51], [66, 100], [67, 51], [67, 100], [68, 31], [68, 51], [68, 100], [69, 4], [69, 51], [69, 91], [69, 100], [69, 155], [70, 60], [70, 71], [70, 80], [71, 40], [71, 60], [71, 71], [71, 80], [72, 31], [72, 71], [73, 20], [75, 11], [76, 40], [77, 4], [77, 20], [79, 31], [79, 120], [79, 123], [79, 137], [79, 154], [80, 11], [81, 11], [82, 4], [84, 40], [84, 111], [85, 4], [85, 31], [87, 57], [87, 65], [87, 70], [87, 76], [87, 83], [87, 88], [87, 89], [88, 128], [88, 148], [88, 149], [90, 40], [90, 47], [90, 96], [91, 47], [91, 96], [91, 155], [92, 11], [92, 47], [92, 96], [92, 155], [93, 155], [94, 20], [95, 6], [95, 7], [95, 8], [95, 31], [95, 44], [95, 45], [95, 121], [95, 122], [95, 123], [95, 143]])

	sep = get_opt_kmean('big_map.png', X, 10)

	write_to_imgf('big_map.png', sep, 'result1.png')

	i = 0
	while True:

		if i >= len(sep):
			break

		if len(sep[i]) > 15:
			'''
			X_label, density = run_kmeans(4, sep[i], 'big_map.png')

			temp_sep = [[] for row in range(4)]

			for j in range(len(sep[i])):
				temp_sep[X_label[j]].append(sep[i][j])
			'''
			temp_sep = get_opt_kmean('big_map.png', sep[i], 10)

			for t in temp_sep:
				sep.append(t)

		i+=1

	write_to_imgf('big_map.png', sep, 'result2.png')
