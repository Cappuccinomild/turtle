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
import tsp
import ast

global path_adj
global image_name


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
    # make adj list
    for i in range(len(X)):
        for j in range(i, len(X)):
            path_adj[i][j] = len(astar(img, X[i], X[j]))

    for i in range(len(X)):
        for j in range(i, len(X)):
            path_adj[j][i] = path_adj[i][j]

    return path_adj


# init
def init_kmean(K, X, img):
    image_name = 'pointed_map.png'
    mean = set()
    X_label = []

    path_adj = init_adj(X, image_name)

    # 초기 점 설정

    # 첫번째 점은 랜덤
    point = random.randint(0, len(X) - 1)

    # 이전에 선택된 점과의 거리를 기준으로 가장 먼 점이 높은 확률로 선정되게 한다

    while len(mean) < K:
        sample_rate = []
        # 0~1까지의 확률을 가진 실수 리스트
        total = sum(path_adj[point])

        for i in range(len(X)):
            sample_rate.append(path_adj[i][point] / total)

        # 선택
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

    # 초기점에따라 군집화
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
            # length = len(astar(img, X[x_i], X[i]))
            if s > length:
                s = length
                label = i

        X_label.append(mean.index(label))

        cnt += 1

    return X_label


# calc G
def calc_G(K, X, img, X_label):
    mean = []
    label_cnt = []

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
                    return np.array([row, col + flag])
            flag = flag + 1
    else:
        return np.array([row, col])


def write_to_img(img_name, X, X_label):
    img = cv2.imread(img_name)
    color_list = []

    while len(color_list) <= max(X_label) + 1:
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
    # X = np.array([[[40,1], [42, 10]], [[10, 34], [21,16], [2, 23]], [[37, 26], [39, 37], [37, 41], [47, 48], [48,12]]])
    # X = np.array([[40,1], [42, 10], [10, 34], [21,16], [2, 23], [37, 26], [39, 37], [37, 41], [47, 48], [48,12]])
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
        cnt = 0
        for x in X:
            s = 9999
            label = 0
            for i in range(len(mean)):
                length = len(astar(img, x, list(mean[i])))
                if s > length:
                    s = length
                    label = i
            density[label] += s ** 2
            temp_label.append(label)
            cnt += 1

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
    for i in range(1, len(inertias) - 1):
        opt.append((inertias[i - 1] - inertias[i]) / (inertias[i] - inertias[i + 1]))

    return opt.index(max(opt)) + 3


def display_tsp(img_name):
    img_origin = cv2.imread(img_name, 2)
    '''
    list_tsp=[]
    for i in range(K):
        list_tsp.append([])
    for i in range(len(X)):
        list_tsp[X_label[i]].append(X[i])
    for i in range(K):
        print(i,"번째: ",list_tsp[i],len(list_tsp[i]))
    '''
    f1 = open("saved_data_k.txt", 'r')
    f2 = open("saved_data_sep.txt", 'r')
    K = int(f1.readline())
    X_label = f2.read()
    f1.close()
    f2.close()
    list_tsp = ast.literal_eval(X_label)
    tsped_point = []
    print(list_tsp)
    route_file = open('route.txt', 'w')
    for i in range(K):
        temp = [];
        img = cv2.imread(img_name)
        color = list(np.random.random(size=3) * 255)

        tsp_path_adj = init_adj(list_tsp[i], img_name)
        '''for j in range(len(tsp_path_adj)):
            print(j,":",tsp_path_adj[j])'''
        r = range(len(tsp_path_adj))
        # Dictionary of distance
        dist = {(m, j): tsp_path_adj[m][j] for m in r for j in r}
        path_len, path = tsp.tsp(r, dist)
        print(path)
        for j in range(len(path)):
            print(j, "번째:", path[j], list_tsp[i][j])
            temp.append(list_tsp[i][j])
        print(temp)
        tsped_point.append(temp)

        for j in range(len(path) - 1):
            print(path[j], path[j + 1])
            route = (astar(img_origin, tuple(list_tsp[i][path[j]]), tuple(list_tsp[i][path[j + 1]])))
            print("route:", route)
            for l, k in route:
                img[k][l] = color
        cv2.imwrite('route%d.png' % (i), img)
    print('출력: '+ str(tsped_point))
    route_file.write(str(tsped_point))
    route_file.close()


def get_opt_kmean(img_name, X, end):
    # 인수 만들기
    args = []
    for i in range(2, end):
        args.append((i, X, img_name))

    # 멀티프로세싱
    print("processing start")
    pool = multiprocessing.Pool(processes=len(args))  # 현재 시스템에서 사용 할 프로세스 개수
    result = pool.map(run_kmeans, args)
    pool.close()
    pool.join()

    # 분산도 구하기
    mean_dist = []
    for density in result:
        mean_dist.append(sum(density[1]))

    e = elbow(mean_dist)

    print("elbow : ", e)

    X_label, density = result[e - 2]  # 0번째의 K는 2이므로 elbow에서 2를 빼서 K를 찾음

    sep = [[] for row in range(e)]

    for i in range(len(X)):
        sep[X_label[i]].append(list(X[i]))

    return sep


if __name__ == '__main__':
    image_name = 'pointed_map.png'
    f2 = open("original_points.txt", 'r')
    f1 = open("moved_points.txt", 'r')
    X = ast.literal_eval(f1.read())
    X_original = ast.literal_eval(f2.read())
    f1.close()
    f2.close()

    # X = np.array(
    #     [[2, 3], [2, 12], [2, 18], [2, 23], [2, 32], [2, 33], [2, 34], [2, 42], [2, 71], [3, 10], [3, 47], [3, 68], [5, 24], [5, 29], [5, 35], [6, 2], [6, 18], [6, 51], [6, 56], [6, 59], [6, 63], [6, 73], [7, 5], [7, 13], [7, 42], [8, 5], [8, 18], [8, 42], [8, 73], [9, 5], [9, 56], [9, 58], [10, 26], [10, 27], [10, 28], [10, 29], [10, 35], [10, 36], [11, 2], [12, 53], [12, 67], [12, 73], [13, 10], [13, 19], [13, 24], [13, 33], [13, 34], [13, 44], [13, 47], [13, 73], [15, 55], [15, 62], [16, 50], [18, 2], [18, 7], [18, 24], [18, 32], [18, 38], [18, 40], [18, 45], [18, 50], [19, 57], [19, 64], [19, 66], [20, 50], [20, 73], [21, 3], [21, 14], [21, 15], [21, 18], [21, 26], [21, 28], [21, 37], [21, 38], [21, 63], [22, 45], [22, 68], [23, 20], [24, 20], [24, 30], [25, 73], [26, 2], [26, 3], [26, 6], [26, 8], [26, 15], [26, 16], [26, 25], [26, 35], [26, 40], [26, 55], [26, 60], [27, 30], [27, 55], [27, 60], [28, 50], [28, 55], [28, 68], [29, 25], [29, 50], [29, 55], [29, 73], [30, 35], [30, 45], [31, 2], [31, 7], [31, 8], [31, 17], [31, 35], [31, 63], [32, 40], [33, 2], [33, 50], [33, 60], [34, 15], [34, 30], [34, 45], [34, 77], [35, 35], [35, 50], [36, 2], [36, 5], [36, 25], [36, 57], [36, 64], [36, 65], [37, 2], [37, 10], [37, 30], [37, 35], [38, 15], [38, 20], [38, 47], [39, 56], [39, 60], [39, 68], [39, 76], [40, 2], [40, 10], [40, 27], [40, 29], [40, 38], [42, 5], [42, 15], [42, 20], [42, 55], [43, 10], [43, 27], [43, 31], [43, 35], [43, 41], [43, 44], [43, 46], [44, 48], [44, 57], [44, 61], [44, 69], [44, 75], [45, 23], [46, 5], [46, 10], [46, 23], [46, 48], [46, 77], [47, 2], [47, 15], [47, 53], [47, 57], [47, 58], [47, 66], [47, 70], [47, 71], [47, 73]])
    sep = get_opt_kmean(image_name, X, 10)

    write_to_imgf(image_name, sep, 'result1.png')

    i = 0
    while True:

        if i >= len(sep):
            break

        if len(sep[i]) > 15:
            '''
            X_label, density = run_kmeans(4, sep[i], image_name)
            temp_sep = [[] for row in range(4)]
            for j in range(len(sep[i])):
                temp_sep[X_label[j]].append(sep[i][j])
            '''
            temp_sep = get_opt_kmean(image_name, sep[i], 10)
            sep.pop(i)

            for t in temp_sep:
                sep.append(t)

        i += 1

    write_to_imgf(image_name, sep, 'result2.png')

    f = open('saved_data_k.txt', 'w')
    f.write(str(len(sep)))
    f.close()

    f = open('saved_data_sep.txt', 'w')
    f.write(str(sep))
    f.close()

    display_tsp(image_name)
