#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import array 
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.linalg import lstsq
from rdp import rdp
import networkx as nx
from KUKA import YouBot
import time
import math
import cv2
from random import randint


def Init(num_landmarks):  # инициализация
    mu_0 = np.full(3 + 2 * num_landmarks, np.nan, dtype=np.float32)  # вектор состояния
    mu_0[:3] = 0
    cov_0 = np.zeros((3 + 2 * num_landmarks, 3 + 2 * num_landmarks), dtype=np.float32)  # ковариационная матрица
    cov_0[np.diag_indices_from(cov_0)] = 100
    cov_0[np.diag_indices(3)] = 0
    cov_0 = (cov_0 + np.transpose(cov_0)) / 2

    return mu_0, cov_0

def PredictionStep(odom):  # шаг предсказания
    global mu
    global cov

    Rx = np.diag([0.05, 0.05, 0.05]).astype(np.float32)
    x_prev, y_prev, theta_prev = mu[:3]
    mu[:3] = odom[:3]  # новый вектор состояния
    delta = np.abs((odom[0] - x_prev) / np.cos(odom[2]))

    # матрица Якоби модели одометрии
    Gx = np.eye(3)
    Gx[0, 2] -= delta * np.sin(odom[2])
    Gx[1, 2] += delta * np.cos(odom[2])
    sigma_xx = cov[:3, :3]
    sigma_xm = cov[:3, 3:]
    sigma = cov
    sigma[:3, :3] = Gx @ sigma_xx @ Gx.T + Rx
    sigma[:3, 3:] = Gx @ sigma_xm
    cov = (sigma + sigma.T) / 2  # новая матрица ковариаций

def CorrectionStep(angle_dict):  # шаг коррекции
    global mu
    global cov

    x, y, theta = mu[:3]

    zs, zs_pred = np.empty(2), np.empty(2)  # истинные и предсказанные наблюдения

    for key, coord_angle in angle_dict.items():
        
        angle_xy_ot = (coord_angle[0] - x, coord_angle[1] - y)

        # Расширение mu и cov, если обнаружены новые ориентиры
        angle_id_max = int(max(angle_dict.keys()))

        if (angle_id_max+1)*2 > len(mu) - 3:
            n = ((angle_id_max+1)*2 - (len(mu) - 3))//2
            ExtendMuAndCov(n)

        if np.isnan(mu[2 * key + 3]):
            # Инициализация в вектор состояния нового ориентира
            mx, my = coord_angle[0], coord_angle[1]
            mu[2 * key + 3:2 * key + 5] = mx, my

        # добавление ориентира в вектор истинных наблюдений
        zs[0] = np.sqrt(angle_xy_ot[0] ** 2 + angle_xy_ot[1] ** 2)
        zs[1] = np.arctan2(angle_xy_ot[1], angle_xy_ot[0]) - theta

        mx, my = mu[2 * key + 3:2 * key + 5]

        # расстояние от робота до метки (вычисляем ожидаемое наблюдение по текущей оценке)
        delta = np.array([mx - x, my - y])
        q = np.dot(delta, delta)
        sqrt_q = np.sqrt(q)

        # добавление ориентира в вектор предсказанных наблюдений
        zs_pred[0] = sqrt_q
        zs_pred[1] = np.arctan2(delta[1], delta[0]) - theta

        # Вычислите якобиан Hi функции измерения H для этого наблюдения
        delta_x, delta_y = delta

        Q = np.identity(2, dtype=np.float32) * 0.03
        G = np.array([[-sqrt_q * delta_x, - sqrt_q * delta_y, 0, sqrt_q * delta_x, sqrt_q * delta_y],
                      [delta_y, - delta_x, - q, - delta_y, delta_x]])
        G = G / q

        nLM = (len(mu) - 3) // 2
        F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
        F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * ((key + 1) - 1))), np.eye(2), np.zeros((2, 2 * nLM - 2 * (key + 1)))))

        F = np.vstack((F1, F2))
        H = G @ F

        # Усиление Калмана
        K = cov @ H.T @ np.linalg.inv(H @ cov @ H.T + Q)

        mu += K @ (zs - zs_pred).T
        I = np.eye(len(mu), dtype=np.float32)
        cov = (I - (K @ H)) @ cov
        cov = (cov + cov.T) / 2

# Расширение mu и cov, если обнаружен новый ориентир
def ExtendMuAndCov(n):
    global mu
    global cov
    value = len(mu)

    mu = np.pad(mu, pad_width=((0, 2 * n)), mode='constant', constant_values=np.nan)
    cov = np.pad(cov, pad_width=((0, 2 * n), (0, 2 * n)), mode='constant')
    np.fill_diagonal(cov[value:len(mu), value:len(mu)], 100.)

# Перевод и фильтрация данных с лидара
def filter_and_transform(odom, lidar):
    rads = np.linspace(float(odom[2]) + np.radians(120 - 240 * 85 / 681), float(odom[2]) - np.radians(120 - 240 * 85 / 681), num=511)

    points_x = np.zeros(511)
    points_y = np.zeros(511)
    points = np.zeros((511, 2))
    
    x = 0.3 * np.cos(odom[2]) + odom[0]
    y = 0.3 * np.sin(odom[2]) + odom[1]
    
    for i in range(511):
        if lidar[i] < 5.6:
            points_x[i] = x + float(lidar[i]) * np.cos(rads[i])
            points_y[i] = y + float(lidar[i]) * np.sin(rads[i])
            points[i] = (points_y[i], points_x[i])    
    
    return points

# Детектирование углов
def detect_angle(points):    
    angle = []
    clusters = []
    line_two = []

    # выделение кривых линий
    dbscan = DBSCAN(eps=0.5, min_samples=10) # настройка параметров кластреризации
    dbscan.fit(points)
    labels = dbscan.labels_
    label_max = max(labels)

    # выделение ключевых точек
    for label in np.unique(labels):
        if label == -1:
            continue
            
        curve = points[labels == label]
        points_rdp = rdp(curve, epsilon=0.2) # настройка параметров алгоритма rdp
        
        if label == 0 and len(points_rdp) == 2:
            line_two.append(points_rdp[1])

        elif label == max(labels) and len(points_rdp) == 2:
            line_two.append(points_rdp[0])

        elif len(points_rdp) == 2 and label != 0 and label != max(labels):
            continue
            
        else: 
            points_rdp_two = rdp(points_rdp, epsilon=0)
            clusters.append(points_rdp[1:len(points_rdp_two) - 1])
    
    if len(line_two) == 2:
        interval = math.sqrt((line_two[1][0] - line_two[0][0])**2 + (line_two[1][1] - line_two[0][1])**2)
        if interval < 1.5:
            clusters.append(line_two)
        
    # определение углов     
    for i, cluster in enumerate(clusters):
            angle.extend(cluster)

    return np.array(angle)

# Идентицикация углов
def identify_angle(angle_arr):
    global count_angle
    global mu
  
    angle_dict = {}

    for coord in angle_arr:

        angle_dict[count_angle] = [coord[0], coord[1]]
        count_angle += 1

        for i in range(3, len(mu) - 1, 2):
            diff_x = abs(coord[0] - mu[i])
            diff_y = abs(coord[1] - mu[i + 1])
            if diff_x < 0.5 and diff_y < 0.5:
                angle_dict[(i - 3) // 2] = [coord[0], coord[1]] 
                count_angle -= 1
                del angle_dict[count_angle]
                break
    
    return angle_dict

# Детектирование перекрестков
def detect_crossroad(angle_arr):
    
    crossroad_arr = []

    dbscan = DBSCAN(eps=1, min_samples=2) # настройка параметров кластреризации
    dbscan.fit(angle_arr)
    labels = dbscan.labels_

    for label in np.unique(labels):
        if label == -1:
            continue
        crossroad = angle_arr[labels == label]
        crossroad_arr.append(crossroad)

    return crossroad_arr

# Поиск ключа по значению в словаре
def find_key(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

# Идентификация перекрестков
def identify_crossroad(crossroad_arr, angle_dict):  
    global G
     
    crossroad_dict = {}
    count_crossroad = G.number_of_nodes()
    
    for crossroad in crossroad_arr:
        keys = []
        
        for angle in crossroad:
            keys.append(find_key(angle_dict, list(angle)))
    
        found_node = [node for node, attr in G.nodes(data=True) if attr['angles_id'] == keys]
        
        if len(found_node) == 0:
            crossroad_dict[count_crossroad] = keys
            count_crossroad += 1
        else:
            crossroad_dict[found_node[0]] = keys
            
    return crossroad_dict

# Поиск ближайшего перекрестка
def find_nearest_crossroad(robot_x, robot_y, angle_dict, crossroad_dict):
    
    min_distance = float('inf')
    nearest_angle = None
    nearest_crossroad = None
    
    for key, angle_coordinates in angle_dict.items():
        distance = math.sqrt((robot_x - angle_coordinates[0])**2 + (robot_y - angle_coordinates[1])**2)

        if distance < min_distance:
            min_distance = distance
            nearest_angle = key
    
    nearest_crossroad = [key for key, angle_key in crossroad_dict.items() if nearest_angle in angle_key]
    
    return nearest_crossroad[0]

# Поиск метки для движения до центра перекрестка
def find_landmark_for_movement(nearest_crossroad, crossroad_dict, angle_dict, robot_x, robot_y, robot_orientation):
    
    angles_keys = crossroad_dict[nearest_crossroad]
    angles_coordinates = []
    
    for key in angles_keys:
        if key in angle_dict:
            coordinates = angle_dict[key]
            angles_coordinates.append(coordinates)
    
    guide = max(angles_coordinates, key=lambda p: math.sqrt((robot_x - p[0])**2 + (robot_y - p[1])**2))
    
    if len(angles_coordinates) == 2:
        diff_x = abs(angles_coordinates[0][0] - angles_coordinates[1][0])
        diff_y = abs(angles_coordinates[0][1] - angles_coordinates[1][1])

        if (robot_orientation == 90 or robot_orientation == -90) and diff_x < 0.1:
            guide = [guide[0]+0.87, guide[1]]
        if (robot_orientation == 0 or robot_orientation == 180 or robot_orientation == -180) and diff_y < 0.1:
            guide = [guide[0], guide[1]+0.87]
            
    return guide

# Рассчет расстояния между роботом и меткой движения
def calculate_distance(guide, robot_x, robot_y):
    
    distance = math.sqrt((guide[0]-robot_x)**2 + (guide[1]-robot_y)**2)
    
    return distance

# Создание узла в графе
def create_node(nearest_crossroad, crossroad_dict, active_edge):
    global G
    global count_edge
    
    count_edge = G.number_of_edges()
    name = G[active_edge[0]][active_edge[1]]["name"]
    G.add_node(name+1, angles_id = crossroad_dict[nearest_crossroad], state = False, corridors = False, dead_end = False)
    if count_edge == 0:
        G.add_edge(0, 1, name = 0, turn = 0, state = 1, robot = 1)
        count_edge += 1
               
# Создание ребер при достижении перекрестка
def create_edge(active_node, data_lidar, width, robot_orientation, points):
    global G
    global count_edge
    
    right_distance =  points[0]
    forward_distance = points[255]
    left_distance = points[510]
 
    if right_distance > width:
        G.add_edge(active_node, count_edge+1, name = count_edge, turn = robot_orientation+90, state = 0, robot = 0)
        G.add_node(count_edge+1, angles_id = None, state = None, corridors = False, dead_end = False)
        count_edge += 1
        
    if forward_distance > width:
        G.add_edge(active_node, count_edge+1, name = count_edge, turn = robot_orientation, state = 0, robot = 0)
        G.add_node(count_edge+1, angles_id = None, state = None, corridors = False, dead_end = False)
        count_edge += 1
        
    if left_distance > width:
        G.add_edge(active_node, count_edge+1, name = count_edge, turn = robot_orientation-90, state = 0, robot = 0)
        G.add_node(count_edge+1, angles_id = None, state = None, corridors = False, dead_end = False)
        count_edge += 1
        
# Поиск пути до ближайшего узла (перекрестка)
def search_way(nodes_false, active_node):
    global G
    target_nodes = nodes_false

    # Нахождение кратчайшего пути до каждого целевого узла
    shortest_paths = {}
    for target in target_nodes:
        shortest_path = nx.shortest_path(G, source=active_node, target=target)
        shortest_paths[target] = shortest_path

    # Выбор самого короткого пути
    min_length = float('inf')
    selected_target = None
    for target, path in shortest_paths.items():
        if len(path) < min_length:
            min_length = len(path)
            selected_target = target
    
    way = shortest_paths[selected_target]

    return way[0]

# Выбор целевого перекрестка
def choice_goal(active_node):
    global G
    target_attribute = 'state'
    target_value = 0

    # получение списка ребер с заданным значением атрибута
    edges_list = [(u, v) for u, v, attributes in G.edges(active_node, data=True) if attributes.get(target_attribute) == target_value]

    # поиск по графу коридоров с 0 и путь до перекрестка:
    if len(edges_list) == 0:

        #глобальный поиск по всему графу
        nodes_false = [node for node, attributes in G.nodes(data=True) if attributes.get('state') == False]
        goal_node = search_way(nodes_false, active_node)
        attribute_turn = G[active_node][goal_node]['turn']

    else:

        max_num_edges = len(edges_list)
        goal = randint(0,  max_num_edges-1)
        goal_edge = edges_list[goal]
        name = G[goal_edge[0]][goal_edge[1]]['name']
        goal_node = name+1
        attribute_turn = G[goal_edge[0]][goal_edge[1]]['turn']

    return goal_node, attribute_turn

# Рассчет поворота робота на перекрестке
def turn_into_corridor(robot_orientation, turn_edge):
    rotation = turn_edge - robot_orientation
    if goal_node - active_node < 0:
        rotation *= -1
    
    return rotation

# Поворот робота
def rotate_robot(rotation, robot_orientation):
    global dead_end
    
    if dead_end:
        robot.move_base(0, 0, 0.5)
        time.sleep(6.10) #для поворота на 180 градусов
        robot.move_base(0, 0, 0)
        dead_end = False
        
    else:
        if rotation == 90:
            robot_orientation += 90
            print("Я поворачиваю направо")
            robot.move_base(0, 0, -0.5)
            time.sleep(3.05) #для поворота на 90 градусов вправо
            robot.move_base(0, 0, 0)
            robot.move_base(0.2, 0, 0)
            time.sleep(1.5)
            robot.move_base(0, 0, 0)
        
        if rotation == -90:
            robot_orientation -= 90
            print("Я поворачиваю налево")
            robot.move_base(0, 0, 0.5)
            time.sleep(3.05) #для поворота на 90 градусов влево
            robot.move_base(0, 0, 0)
            robot.move_base(0.2, 0, 0)
            time.sleep(1.5)
            robot.move_base(0, 0, 0)

        if rotation == 0:
            print("Прямо")
            robot.move_base(0.2, 0, 0)
            time.sleep(1.5)
            robot.move_base(0, 0, 0)
    
    return robot_orientation

# Определение тупика    
def is_deadend(points, width):
        
    right_distance =  points[0]
    forward_distance = points[255]
    left_distance = points[510]
 
    if forward_distance < width/2 + 0.3 and left_distance < width/2 + 0.3 and right_distance < width/2 + 0.3:
        dead_end = True
    else:
        dead_end = False
    
    return dead_end

#Установка состояния узлов
def check_node_state(active_node):
    global G

    for neighbor in G.neighbors(active_node):
        edge_attributes = G.edges[(active_node, neighbor)]
        state = edge_attributes.get('state')

        if state == 0:
            break
        else:
            G.nodes[active_node]['state'] = True
            
            
robot = YouBot('192.168.88.21', ros=False, camera_enable=False)
count_angle = 0
count_edge = 0
robot_orientation = 0
guide = None
active_node = 0
G = nx.Graph()
G.add_node(count_edge, angles_id = None, state = False, corridors = True, dead_end = False)
G.add_edge(active_node, count_edge+1, name = count_edge, turn = robot_orientation, state = 1, robot = 1)
G.add_node(count_edge+1, angles_id = None, state = None, corridors = False, dead_end = False)
distance = 1.58
width = 1
active_edge = (0, 1)
forward = 0
angle_dict = {}
dead_end = False
mu, cov = Init(0)


while any(data['state'] == False for _, data in G.nodes(data = True)):
    
    data = robot.lidar
    data_odom = data[0]
    data_lidar = data[1]
    data_lidar = data_lidar[85:-85]
    forward = data_lidar[255]

    points_kuka = filter_and_transform(data_odom, data_lidar)
    angle_arr = detect_angle(points_kuka)
    angle_dict = identify_angle(angle_arr)
    PredictionStep(data_odom)
    CorrectionStep(angle_dict)
    
    if guide == None:
        crossroad_arr = detect_crossroad(angle_arr)
        crossroad_dict = identify_crossroad(crossroad_arr, angle_dict)
        robot_x = mu[1]
        robot_y = mu[0]
        nearest_crossroad = find_nearest_crossroad(robot_x, robot_y, angle_dict, crossroad_dict)
        if G.has_node(active_edge[1]):
            create_node(nearest_crossroad, crossroad_dict, active_edge)
        else:
            if G.nodes[active_edge[1]]['state'] == None:
                create_node(nearest_crossroad, crossroad_dict, active_edge)
        guide = find_landmark_for_movement(nearest_crossroad, crossroad_dict, angle_dict, robot_x, robot_y, robot_orientation)
        robot.move_base(0.2, 0, 0)
    
    robot_x = mu[1]
    robot_y = mu[0]
    distance = calculate_distance(guide, robot_x, robot_y)

    if distance > 0.8:
        continue
    
    else:  
        robot.move_base(0, 0, 0)
        guide = None

        #Смена состояния графа
        G.edges[active_edge]['state'] = 2
        G.edges[active_edge]['robot'] = 0
        check_node_state(active_node)

        active_node = active_edge[1] 

        if G.nodes[active_node]['corridors'] == False:
            create_edge(active_node, data_lidar, width, robot_orientation, data_lidar)

        dead_end = is_deadend(data_lidar, width)

        if dead_end:
                #обновление атрибутов узла
                G.add_node(active_node, state = True, corridors = True, dead_end = True)
                G.edges[active_edge]["state"] = 1

                if robot_orientation > 0:
                    robot_orientation -= 180
                else:
                    robot_orientation += 180

                goal_node = list(G.neighbors(active_node))[0]
                active_edge = (active_node, goal_node)

                G.edges[active_edge]["robot"] = 1

                robot_orientation = rotate_robot(rotation, robot_orientation) # поворот, если тупик
        else:
            goal_node, turn_edge = choice_goal(active_node)

            active_edge = (active_node, goal_node)

            #обновление состояния узла
            G.edges[active_edge]["robot"] = 1
            G.edges[active_edge]["state"] = 1
            check_node_state(active_node)
            
            rotation = turn_into_corridor(robot_orientation, turn_edge)
            robot_orientation = rotate_robot(rotation, robot_orientation)
            
robot.move_base(0, 0, 0)
print("The end")

