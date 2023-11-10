import math
import os
import time
import threading

import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from PIL import Image
import cv2

#############################################
################# Functions #################
#############################################

def convert_theta_5(theta_0, theta_5):
    """theta_0에 따라 달라지는 theta_5를 계산

    Args:
        theta_0 (float): theta 0 [radian] 또는 [degree]
        theta_5 (float): theta 5 [radian] 또는 [degree]

    Returns:
        float: 조정된 theta 5를 반환
    """
    
    cvtd_theta_5 = theta_5 - theta_0
    
    return -(cvtd_theta_5)

def pixel2length(pixel):
    """픽셀 단위를 [mm] 단위로 변환

    Args:
        pixel (int): 변환할 픽셀 값

    Returns:
        float: 변환된 픽셀의 [mm]
    """
    
    length = pixel * 1.8
    
    return length

def get_radius(center):
    """매니퓰레이터의 중심에서 물체와의 거리를 계산

    Args:
        center (int[]): 물체 중심의 픽셀 값

    Returns:
        float: 매니퓰레이터 중심에서 물체와의 거리 [mm]
    """
    
    buffer = 15
    # buffer = 0
    
    home = [352, 292]
    
    x = abs(center[0] - home[0])
    y = abs(center[1] - home[1])
    
    radius = pixel2length(math.sqrt(x**2 + y**2) - buffer)
    
    return radius

def get_obj_height(depth):
    """그리퍼 파지를 위한 높이 계산

    Args:
        depth (int): 물체와 카메라와의 거리 [mm]

    Returns:
        int: 그리퍼의 파지를 위한 높이 [mm]
    """
    
    std_depth = 1075
    obj_height = std_depth - depth
    
    buffer = 70
    grasp_height = (obj_height / 2) + buffer
    
    # 안전장치
    if (depth > 1070):
        grasp_height = 150
    
    return grasp_height

def get_theta_0(center):
    """물체의 중심 좌표를 기준으로 theta_0를 계산

    Args:
        center (int[]): 물체의 중앙 좌표 픽셀

    Returns:
        float: Joint_0의 각 [radian]
    """

    home = [352, 292]
    
    x = abs(center[0] - home[0])
    y = abs(center[1] - home[1])
    
    theta_0 = math.atan(x/y)
    
    # 부호 결정
    if (center[0] - home[0]) > 0:
        theta_0 = -theta_0
    
    return theta_0

def makeFit(theta_0, theta_1, theta_2, theta_4, theta_5) :
    """역기구학 결과를 Open-Manipulator-P에 맞도록 변환

    Args:
        theta_0 (float): Joint_0의 각[radian]
        theta_1 (float): Joint_1의 각[radian]
        theta_2 (float): Joint_2의 각[radian]
        theta_4 (float): Joint_4의 각[radian]
        theta_5 (float): Joint_5의 각[radian]

    Returns:
        float[]: 변환된 Joint의 각들이 있는 리스트 [radian]
    """
    
    cvtd_Theta_0 = theta_0
    # cvtd_Theta_0 = theta_0 + math.radians(math.asin(21.5/float(radius)))
    cvtd_Theta_1 = math.radians(83.52) - theta_1
    cvtd_Theta_2 = math.radians(-38.52) - theta_2
    cvtd_Theta_4 = - theta_4
    cvtd_theta_5 = theta_5
    
    result = [cvtd_Theta_0, cvtd_Theta_1, cvtd_Theta_2, cvtd_Theta_4, cvtd_theta_5]
    
    return result

def inverseKinematics(radius, height) :
    """역기구학을 통해 각 관절의 각을 계산

    Args:
        radius (float): 물체 중심과 매니퓰레이터 중심과의 거리[mm] 즉, 목표지점의 너비[mm]
        height (float): 환경을 고려한 물체의 높이[mm] 즉, 목표지점의 높이[mm]

    Returns:
        float: 각 관절의 각 [radian]
    """
    
    L1 = 265.69
    L2 = 259.73

    hypotenuse = (radius**2) + (height**2)
    beta = math.atan2(height, radius)
    
    # 해 존재 유무 검사
    if (math.sqrt(hypotenuse) > L1 + L2):
        print("no answers")
        return None
        
    domain = (hypotenuse + (L1**2) - (L2**2)) / float(2 * L1 * math.sqrt(hypotenuse))
    if (-1 <= domain and domain <= 1):
        psi = math.acos(domain)
    else :
        psi = 0

    # A Solution Set
    theta_1A = 0
    theta_2A = 0
    theta_4A = 0

    theta_2A = math.acos((hypotenuse - (L1**2) - (L2**2)) / float(2 * L1 * L2))

    if (theta_2A < 0) :
        theta_1A = beta + psi
    else :
        theta_1A = beta - psi
        
    theta_4A = math.radians(-90) - theta_1A - theta_2A

    # B Solution Set
    theta_1B = 0
    theta_2B = 0
    theta_4B = 0

    theta_2B = - theta_2A

    if (theta_2B < 0) :
        theta_1B = beta + psi
    else :
        theta_1B = beta - psi
        
    theta_4B = math.radians(-90) - theta_1B - theta_2B

    # Choose Theta Set
    theta_1 = 0
    theta_2 = 0
    theta_4 = 0

    if (abs(theta_4A) <= (math.pi/2)) :
        theta_1 = theta_1A
        theta_2 = theta_2A
        theta_4 = theta_4A
    else :
        theta_1 = theta_1B
        theta_2 = theta_2B
        theta_4 = theta_4B
    
    return theta_1, theta_2, theta_4

def bool2coor(input):
    """Mask를 좌표 배열로 변환

    Args:
        input (bool[][]): 세그멘테이션 마스크 또는 Edge detection된 세그멘테이션 마스크

    Returns:
        int[]: 마스크의 좌표 값 [x, y]를 갖는 ndArray
    """
    
    coorList = []

    #  i 행
    for i in range (0, len(input)):
        # j 열
        for j in range (0, int(input.size/len(input))):
            if (input[i][j]):
                coorList.append([j, i])

    coorArray = np.array(coorList) 
    
    return coorArray

def edge_detect(input):
    """물체의 테두리만 표현하도록 함

    Args:
        input (bool[][]): 세그멘테이션 마스크

    Returns:
        int[][]: 자신을 기준으로 상하좌우가 전부 1인 픽셀의 좌표들의 배열, [x, y]가 아닌 [행, 열]
    """
    
    coorList = []
    
    #  i 행
    for i in range (1, len(input) - 1):
        # j 열
        for j in range (1, int(input.size/len(input)) - 1):
            if ((input[i + 1][j]) and (input[i - 1][j]) and (input[i][j + 1]) and (input[i][j - 1])):
                coorList.append([i, j])
    coorArray = np.array(coorList)

    # 해당하는 픽셀들의 값을 0으로 변경
    for i in coorArray:
        input[i[0]][i[1]] = 0
    
    return input

def centerCrossAngle(obj_mask, steps, bbox):
    """축 변환 회귀 직선 검출 알고리즘

    Args:
        obj_mask (bool[][]): 세그멘테이션 마스크
        steps (float, int): SSE 검출 단계
        bbox (int[]): 물체의 바운딩 박스

    Returns:
        float: 축 변환 회귀 직선 기울기 [degree]
    """
    
    step = 360 / steps
    
    # 물체의 중앙 좌표 구하기
    center = [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)]
    
    SSE_List = []
    coorArray = bool2coor(obj_mask)
    
    for i in range (0, int(steps)):
        SSE = 0
        theta = step * i
        
        if 45 <= theta and theta <= 135 or 225 <= theta and theta <= 315:
            if theta < 180:
                # 1, 2 사분면
                cvtd_theta = 90 - theta
            else:
                # 3, 4 사분면
                cvtd_theta = 270 - theta
            slope = math.tan(math.radians(cvtd_theta))
            for x in range(len(obj_mask)):
                y_ = slope * x - (slope * center[1]) + center[0]
                for coor in (coorArray):
                    if coor[1] == x:
                        SSE += ((abs(y_) - abs(coor[0]))**2)

        else:
            slope = math.tan(math.radians(theta))
            for x in range (int(obj_mask.size/len(obj_mask))):
                y_ = slope * x -(slope * center[0]) + center[1]
                for coor in (coorArray):
                    if coor[0] == x:
                        SSE += ((abs(y_) - abs(coor[1]))**2)
            
        SSE_List.append(SSE)

    min_angle_idx = SSE_List.index(min(SSE_List))
    
    return step * min_angle_idx
    
def find_area(input):
    """세그멘테이션 마스크 영역 넓이 계산

    Args:
        input (bool[][]): 세그멘테이션 마스크

    Returns:
        int: 세그멘테이션 마스크 영역 넓이
    """
    
    obj_area = 0
    
    #  i 행
    for i in range (0, len(input)):
        # j 열
        for j in range (0, int(input.size/len(input))):
            if (input[i][j]):
                obj_area += 1
    return obj_area

def rotateObj(input, steps):
    """객체 회전 알고리즘

    Args:
        input (bool[][]): 세그멘테이션 마스크
        steps (float, int): 회전 기울기 검출 단계

    Returns:
        float: 객체 형상 기울기 [radian]
    """
    
    # 물체의 넓이 구하는 과정
    obj_area = find_area(input)
    
    # 엣지 디텍션
    edge_list = edge_detect(input)
    
    # 불리언 배열을 좌표 배열로 변환
    coor_array = bool2coor(edge_list)
    
    # coor_array의 좌표들을 좌표계 변환
    cvtd_coor_array = np.zeros(shape=(steps, len(coor_array), 2))
    
    bowl = []
    
    for i in range(0, steps):
        bowl.clear()
        step = 360 / steps

        rotate_matrix = np.array([[math.cos(math.radians(step * i)), -(math.sin(math.radians(step * i)))],
                                  [math.sin(math.radians(step * i)),  (math.cos(math.radians(step * i)))]])

        for j in coor_array:
            p_ = rotate_matrix.dot(np.transpose(j))
            bowl.append([int(p_[0]), int(p_[1])])

        # cvtd_coor_array[i]는 step * i 만큼 회전된 좌표(int)들의 배열
        cvtd_coor_array[i] = np.array(bowl)

    # step * i 마다 바운딩 박스 좌표를 찾는 과정
    area_list = []
    
    # x가 길면 1, y가 길면 0
    orient_hint_list = []
    
    for i in range(0, steps):
        x_max = np.max(cvtd_coor_array[i][:,0])
        y_max = np.max(cvtd_coor_array[i][:,1])

        x_min = np.min(cvtd_coor_array[i][:,0])
        y_min = np.min(cvtd_coor_array[i][:,1])

        box_area = (x_max - x_min) * (y_max - y_min)
        
        #  step * i 마다 넓이 차를 저장
        area_list.append(box_area - obj_area)

        #  step * i 마다 긴 부분 힌트를 저장
        if (x_max - x_min) > (y_max - y_min):
            orient_hint_list.append(1)
        else :
            orient_hint_list.append(0)
            
    area_array = np.array(area_list)
    
    min_idx = np.argmin(area_array)
    
    # 각도(radian)를 반환
    if (orient_hint_list[min_idx]):
        return math.radians(int(step * np.argmin(area_array)) + 90)
    else :
        return math.radians(int(step * np.argmin(area_array)))

def move(to_manipulator):
    """Open-Manipulator-P에게 이동 명령 하달

    Args:
        to_manipulator (float[]): 목표 관절 각도의 리스트
    """
    
    for i in range(5) :
        to_manipulator[i] = int(to_manipulator[i]*1000)

    cmd = "rosrun open_manipulator_p_teleop client"

    for i in to_manipulator:
        cmd += " " + str(i)
    
    os.system(cmd)
    
def gripper_close():
    """Open-Manipulator-P의 그리퍼를 닫는 명령 하달
    """
    
    cmd = "rosservice call grab 1"
    os.system(cmd)

def gripper_open():
    """Open-Manipulator-P의 그리퍼를  여는 명령 하달
    """
    
    cmd = "rosservice call grab 0"
    os.system(cmd)

def classify(class_name):
    """Class에 따라 분류 위치를 결정

    Args:
        class_name (string): 추론의 결과 중 클래스 이름

    Returns:
        float, float: 물체가 놓여져야하는 위치의 각 [radian], 거리 [mm]
    """
    
    print("class name : " + class_name)
    
    if (class_name == 'can'):
        return math.radians(-75), 400
    elif (class_name == 'plastic'):
        return math.radians(105), 400
    elif (class_name == 'glass'):
        return math.radians(75), 400
    elif (class_name == 'paperpack'):
        return math.radians(-105), 400
    else :
        return math.radians(105), 400

#############################################
################## Classes ##################
#############################################
class Segmentation():
    """세그멘테이션을 진행하는 클래스
    """
    def __init__(self):
        self.model = YOLO('/root/weight.pt')

    def execute(self, input_image):
        
        output = self.model(input_image)
        
        reachable_obj_idx_list = []
        flag = False
        
        # output is the number of pictures
        if output and len(output) != 0:
            
            for instances in output:
                detection_cnt = instances.boxes.shape[0]
                
                if (detection_cnt == 0):
                    return None
                
                for i in range (detection_cnt):
                    bbox = instances.boxes.xyxy[i].cpu().numpy()
                    center = [int((bbox[0] + bbox[2]) / 2) , int((bbox[1] + bbox[3]) / 2)]    
                    
                    # limit space
                    if 230 < center[0] and center[0] < 480:
                        if 15 < center[1] and center[1] < 225: 
                            flag = True
                    
                    if flag:
                        reachable_obj_idx_list.append(i)
                        flag = False
                
                if len(reachable_obj_idx_list) == 0:
                    return None
                
                # choice[0] : idx
                # choice[1] : conf
                choice = [-1, 0]
                
                for i in reachable_obj_idx_list:
                    
                    if choice[1] < float(instances.boxes.conf[i].item()):
                        choice[0] = i
                        choice[1] = float(instances.boxes.conf[i].item())
                    
                max_index = choice[0]
                
                if choice[0] == -1:
                    return None
                
                # TODO check order 1, 2
                bbox = instances.boxes.xyxy[max_index].cpu().numpy()
                center = [int((bbox[0] + bbox[2]) / 2) , int((bbox[1] + bbox[3]) / 2)]
                
                # angle 계산
                mask = instances.masks.data[max_index].cpu().numpy()
                
                # 확인용 사진 저장
                # segmented_image = Image.fromarray(np.uint8(np.where(mask[..., None], input_image, [0, 0, 0])))
                # segmented_image.save('/root/yolov8/output.png', format='PNG')
                
                theta_5 = rotateObj(mask, 36)
        
                # class_id
                class_name = instances.names[int(instances.boxes.cls[max_index].item())]
        
        # return [[center[0] - align_x, center[1] - align_y], theta_5, class_name]
        return [[center[0], center[1]], theta_5, class_name]
    

# multi-threading setting
lock = threading.Lock()

class Depth_Camera():
    """Intel Realsense2 시리즈 중 D435, D415를 활용하여 이미지, 거리 맵 획득
    """
    
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        self.align_to = None

        context = rs.context()
        connect_device = None
        if context.devices[0].get_info(rs.camera_info.name).lower() != 'platform camera':
            connect_device = context.devices[0].get_info(rs.camera_info.serial_number)

        print(" > Serial number : {}".format(connect_device))
        self.config.enable_device(connect_device)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)

    def __del__(self):
        print("Collecting process is done.\n")

    def execute(self):
        try:
            self.pipeline.start(self.config)
        except:
            return
        
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        try:
            while(True):
                lock.acquire()
                
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                depth_info = depth_frame.as_depth_frame()
                self.bgr_color_image = np.asanyarray(color_frame.get_data())
                self.rgb_color_image = cv2.cvtColor(self.bgr_color_image, cv2.COLOR_BGR2RGB)
                # cv2.imshow('color', self.color_image)
                # cv2.waitKey(100)
                # cv2.destroyAllWindows()
                self.depth_image = depth_info
                
                lock.release()
            
        finally:
            self.pipeline.stop()
    
    def get(self):
        return self.rgb_color_image, self.depth_image
    
#############################################
################### Main ####################
#############################################

# Connect Camera
camera = Depth_Camera()

# Connect Manipulator
# os.system("roscore")
# os.system("roslaunch open_manipulator_p_controller open_manipulator_p_controller.launch with_gripper:=true")
# os.system("roslaunch open_manipulator_p_teleop open_manipulator_p_teleop_keyboard.launch with_gripper:=true")

# Threading Setting
camera_thread = threading.Thread(target=camera.execute)
camera_thread.daemon = True
camera_thread.start()

# Initial Set
segmentation = Segmentation()

# Initial Setting
home = [0, 0, -math.pi/4, 0, 0]
move(home)
gripper_open()
time.sleep(6)
color_image, depth_image = camera.get()
inf_result = segmentation.execute(color_image)

while(True):
    
    # Limit Workspace
    if (inf_result is None):
        # Initializing again
        gripper_open()
        # stopby = [0, 0, 0, 0, 0]
        # move(stopby)
        # time.sleep(2)
        home = [0, 0, -math.pi/4, 0, 0]
        move(home)
        time.sleep(2)
        color_image, depth_image = camera.get()
        inf_result = segmentation.execute(color_image)
        continue

    ### Pick-and-Place ###

    # calculate standards
    obj_center = inf_result[0]
    theta_5 = inf_result[1]
    theta_0 = get_theta_0(obj_center)
    class_name = inf_result[2]
    depth = int(round((depth_image.get_distance(obj_center[0], obj_center[1]) * 1000), 1))
    radius = get_radius(obj_center)
    height = get_obj_height(depth)
    cvtd_theta_5 = convert_theta_5(theta_0, theta_5)
    
    # Pick Process 1 : move to top of object, in 2s
    temp_height = height * 1.5
    if (inverseKinematics(radius, temp_height) is None):
        inf_result = None
        continue
    theta_1, theta_2, theta_4 = inverseKinematics(radius, temp_height)
    if (abs(theta_4) >= (math.pi/2)) :
        print()
    to_manipulator = makeFit(theta_0, theta_1, theta_2, theta_4, cvtd_theta_5)
    move(to_manipulator)
    time.sleep(2)
    
    # Pick Process 2 : move to object, open gripper, in 2s
    if (inverseKinematics(radius, height + 12) is None):
        inf_result = None
        continue
    theta_1, theta_2, theta_4 = inverseKinematics(radius, height + 12)
    if (abs(theta_4) >= (math.pi/2)) :
        print()
    to_manipulator = makeFit(theta_0, theta_1, theta_2, theta_4, cvtd_theta_5)
    move(to_manipulator)
    time.sleep(2)
    
    # Pick Process 3 : close gripper, in 1s
    gripper_close()
    time.sleep(1)
    
    # Pick Process 4 : move upward, in 2s
    temp_height = 220
    
    while(True):
        if (inverseKinematics(radius, temp_height) is None):
            temp_height -= 1
        else :
            break
    
    theta_1, theta_2, theta_4 = inverseKinematics(radius, temp_height)
    
    if (abs(theta_4) >= (math.pi/2)) :
        print()
    
    to_manipulator = makeFit(theta_0, theta_1, theta_2, theta_4, cvtd_theta_5)
    move(to_manipulator)
    time.sleep(2)
    
    # Place Process 1 : move to top of trash can, in 2s
    cvtd_theta_0, radius = classify(class_name)
    if (inverseKinematics(radius, temp_height) is None):
        inf_result = None
        continue
    theta_1, theta_2, theta_4 = inverseKinematics(radius, temp_height)
    if (abs(theta_4) >= (math.pi/2)) :
        print()
    to_manipulator = makeFit(cvtd_theta_0, theta_1, theta_2, theta_4, cvtd_theta_5)
    move(to_manipulator)
    time.sleep(2)
    
    # Pick Process 6 : open gripper, in 0.8s
    gripper_open()
    
    # Take picture
    color_image, depth_image = camera.get()
    
    # Instance Segmentation
    inf_result = segmentation.execute(color_image)
    
    time.sleep(0.5)