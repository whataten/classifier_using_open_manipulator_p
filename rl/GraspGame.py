import numpy as np
import math
from PIL import Image

MOVE = 4

class GraspGame:
    
    def __init__(self):
        # initialMap : int[400][400]
        # currentMap : int[400][400]
        # initialPos : [[x1, y1], [x2, y2]]
        # currentPos : [[x1, y1], [x2, y2]]
        # coorList : int[열, 행] , int[x, y]
        
        self.max = 0
        self.move = 0
        
        img = Image.open('').convert('L')
        img_array = np.array(img)

        background = np.zeros((400, 400), dtype=int)

        img_height, img_width = img_array.shape
        background_height, background_width = background.shape

        start_raw = (background_height - img_height) // 2
        start_col = (background_width - img_width) // 2

        background[start_raw:start_raw+img_height, start_col:start_col+img_width] = img_array
        
        self.initialMap = background
        self.currentMap = background
        
        self.initialPos = [[199, 199],[201, 201]]
        self.currentPos = [[199, 199],[201, 201]]
        
        self.coorList = []
        
        #  i 행
        for i in range (0, len(background)):
            # j 열
            for j in range (0, int(background.size/len(background))):
                if (background[i][j]):
                    self.coorList.append([j, i])
    
    def reset(self):
        self.currentMap = self.initialMap
        self.currentPos = self.initialPos
        self.max = 0
        self.move = 0
        
    def up(self):
        self.move += MOVE
        if ((self.currentPos[0][1] > 2) and (self.currentPos[1][1] > 2)):
            self.currentPos[0][1] = self.currentPos[0][1] - 1
            self.currentPos[1][1] = self.currentPos[1][1] - 1
        
            
    def down(self):
        self.move += MOVE
        if ((self.currentPos[0][1] < 398) and (self.currentPos[1][1] < 398)):
            self.currentPos[0][1] = self.currentPos[0][1] + 1
            self.currentPos[1][1] = self.currentPos[1][1] + 1
            
    def left(self):
        self.move += MOVE
        if ((self.currentPos[0][0] > 2) and (self.currentPos[1][0]) > 2):
            self.currentPos[0][0] = self.currentPos[0][0] - 1
            self.currentPos[1][0] = self.currentPos[1][0] - 1
            
    def right(self):
        self.move += MOVE
        if ((self.currentPos[0][0] < 398) and (self.currentPos[1][0]) < 398):
            self.currentPos[0][0] = self.currentPos[0][0] + 1
            self.currentPos[1][0] = self.currentPos[1][0] + 1
    
    def clockwise(self, counter):
        # 조건문의 기준이 되는 point : currentPos[0]
        # center : 두 점의 중간 점
        # counter : 방향을 결정하는 변수로 -1일 때 반시계방향, 그 외에는 시계 방향
        
        self.move += MOVE
        
        center = [int((self.currentPos[0][0] + self.currentPos[1][0]) / 2), int((self.currentPos[0][1] + self.currentPos[1][1]) / 2)]
        
        # currentPos[0]이 1 사분면에 위치할 때
        if ((self.currentPos[0][0] > center[0]) and (self.currentPos[0][1] < center[1])):
            if (counter == -1):
                if (self.currentPos[0][0] > 2 and self.currentPos[0][1] > 2 and self.currentPos[1][0] < 398 and self.currentPos[1][1] < 398):
                    
                    self.currentPos[0][0] = self.currentPos[0][0] - 1
                    self.currentPos[0][1] = self.currentPos[0][1] - 1
                
                    self.currentPos[1][0] = self.currentPos[1][0] + 1
                    self.currentPos[1][1] = self.currentPos[1][1] + 1
            else :
                if (self.currentPos[0][0] < 398 and self.currentPos[0][1] < 398 and self.currentPos[1][0] > 2 and self.currentPos[1][1] > 2):
                    
                    self.currentPos[0][0] = self.currentPos[0][0] + 1
                    self.currentPos[0][1] = self.currentPos[0][1] + 1
                    
                    self.currentPos[1][0] = self.currentPos[1][0] - 1
                    self.currentPos[1][1] = self.currentPos[1][1] - 1
        
        # currentPos[0]가 2 사분면에 위치할 때
        elif ((self.currentPos[0][0] < center[0]) and (self.currentPos[0][1]  < center[1])):
            if (counter == -1):
                if (self.currentPos[0][0] > 2 and self.currentPos[0][1] < 398 and self.currentPos[1][0] < 398 and self.currentPos[1][1] > 2):
                    
                    self.currentPos[0][0] = self.currentPos[0][0] - 1
                    self.currentPos[0][1] = self.currentPos[0][1] + 1
                    
                    self.currentPos[1][0] = self.currentPos[1][0] + 1
                    self.currentPos[1][1] = self.currentPos[1][1] - 1
            else :
                if (self.currentPos[0][0] < 398 and self.currentPos[0][1] > 2 and self.currentPos[1][0] > 2 and self.currentPos[1][1] < 398):
                    
                    self.currentPos[0][0] = self.currentPos[0][0] + 1
                    self.currentPos[0][1] = self.currentPos[0][1] - 1
                    
                    self.currentPos[1][0] = self.currentPos[1][0] - 1
                    self.currentPos[1][1] = self.currentPos[1][1] + 1
        
        # currentPos[0]가 3 사분면에 위치할 때
        elif ((self.currentPos[0][0] < center[0]) and (self.currentPos[0][1] > center[1])):
            if (counter == -1):
                if (self.currentPos[0][0] < 398 and self.currentPos[0][1] < 398 and self.currentPos[1][0] > 2 and self.currentPos[1][1] > 2):
                    
                    self.currentPos[0][0] = self.currentPos[0][0] + 1
                    self.currentPos[0][1] = self.currentPos[0][1] + 1
                    
                    self.currentPos[1][0] = self.currentPos[1][0] - 1
                    self.currentPos[1][1] = self.currentPos[1][1] - 1
            else :
                if (self.currentPos[0][0] > 2 and self.currentPos[0][1] > 2 and self.currentPos[1][0] < 398 and self.currentPos[1][1] < 398):
                    
                    self.currentPos[0][0] = self.currentPos[0][0] - 1
                    self.currentPos[0][1] = self.currentPos[0][1] - 1
                    
                    self.currentPos[1][0] = self.currentPos[1][0] + 1
                    self.currentPos[1][1] = self.currentPos[1][1] + 1
        
        # currentPos[0]가 4 사분면에 위치할 때
        elif ((self.currentPos[0][0] > center[0]) and (self.currentPos[0][1] > center[1])):
            if (counter == -1):
                if (self.currentPos[0][0] < 398 and self.currentPos[0][1] > 2 and self.currentPos[1][0] > 2 and self.currentPos[1][1] < 398):
                    
                    self.currentPos[0][0] = self.currentPos[0][0] + 1
                    self.currentPos[0][1] = self.currentPos[0][1] - 1
                    
                    self.currentPos[1][0] = self.currentPos[1][0] - 1
                    self.currentPos[1][1] = self.currentPos[1][1] + 1
            else :
                if (self.currentPos[0][0] > 2 and self.currentPos[0][1] < 398 and self.currentPos[1][0] < 398 and self.currentPos[1][1] > 2):
                    
                    self.currentPos[0][0] = self.currentPos[0][0] - 1
                    self.currentPos[0][1] = self.currentPos[0][1] + 1
                    
                    self.currentPos[1][0] = self.currentPos[1][0] + 1
                    self.currentPos[1][1] = self.currentPos[1][1] - 1
        
        # currentPos가 x 축상에 위치할 때
        elif ((self.currentPos[0][0] == center[0]) and (self.currentPos[1][0] == center[0])):
            if not (center[1] < 2 or center[1] > 398):
                if (counter == -1):
                        if (self.currentPos[0][1] < center[1]):
                            self.currentPos[0][0] = self.currentPos[0][0] - 1
                            self.currentPos[1][0] = self.currentPos[1][0] + 1
                        else :
                            self.currentPos[0][0] = self.currentPos[0][0] + 1
                            self.currentPos[1][0] = self.currentPos[1][0] - 1
                        
                else :
                    if (self.currentPos[0][1] < center[1]):
                        self.currentPos[0][0] = self.currentPos[0][0] + 1
                        self.currentPos[1][0] = self.currentPos[1][0] - 1
                    else :
                        self.currentPos[0][0] = self.currentPos[0][0] - 1
                        self.currentPos[1][0] = self.currentPos[1][0] + 1
                
        # currentPos가 y 축상에 위치할 때
        elif ((self.currentPos[0][1] == center[1]) and (self.currentPos[1][1] == center[1])):
            if not (center[0] < 2 or center[0] > 398):
                if (counter == -1):
                    if (self.currentPos[0][0] < center[0]):
                        self.currentPos[0][1] = self.currentPos[0][1] + 1
                        self.currentPos[1][1] = self.currentPos[1][1] - 1
                    else :
                        self.currentPos[0][1] = self.currentPos[0][1] - 1
                        self.currentPos[1][1] = self.currentPos[1][1] + 1
                else :
                    if (self.currentPos[0][0] < center[0]):
                        self.currentPos[0][1] = self.currentPos[0][1] - 1
                        self.currentPos[1][1] = self.currentPos[1][1] + 1
                    else :
                        self.currentPos[0][1] = self.currentPos[0][1] + 1
                        self.currentPos[1][1] = self.currentPos[1][1] - 1
                    
    def approach(self):
        # center : 두 점의 중간 점
        # operator : +1, -1 둘 중 하나를 가지며 픽셀 이동을 위한 값
        # length : 두 점 사이의 거리
        
        self.move += MOVE
        
        center = [int((self.currentPos[0][0] + self.currentPos[1][0]) / 2), int((self.currentPos[0][1] + self.currentPos[1][1]) / 2)]
        operator = 1
        length = math.sqrt((self.currentPos[0][0] - self.currentPos[1][0])**2 + (self.currentPos[0][1] - self.currentPos[1][1])**2)

        # 최소 거리 2
        if (length > 2):
            # 일직선 상(축)에 위치한 경우
            if (self.currentPos[0][0] == self.currentPos[1][0]):
                if (self.currentPos[0][1] < self.currentPos[1][1]):
                    self.currentPos[0][1] = self.currentPos[0][1] + 1
                    self.currentPos[1][1] = self.currentPos[1][1] - 1
                else :
                    self.currentPos[0][0] = self.currentPos[0][0] - 1
                    self.currentPos[1][0] = self.currentPos[1][0] + 1
            elif (self.currentPos[0][1] == self.currentPos[1][1]):
                if (self.currentPos[0][0] < self.currentPos[1][0]):
                    self.currentPos[0][0] = self.currentPos[0][0] + 1
                    self.currentPos[1][0] = self.currentPos[1][0] - 1
                else :
                    self.currentPos[0][0] = self.currentPos[0][0] - 1
                    self.currentPos[1][0] = self.currentPos[1][0] + 1

            # 일직선 상(축)에 위치하지 않은 경우
            else :
                # 기준 : currentPos[0][0]
                operator = 1 if (center[0] - self.currentPos[0][0] > 0) else -1
                self.currentPos[0][0] = self.currentPos[0][0] + operator
                
                # 기준 : currentPos[0][1]
                operator = 1 if (center[1] - self.currentPos[0][1] > 0) else -1
                self.currentPos[0][1] = self.currentPos[0][1] + operator
                
                # 기준 : currentPos[1][0]
                operator = 1 if (center[0] - self.currentPos[1][0] > 0) else -1
                self.currentPos[1][0] = self.currentPos[1][0] + operator
                
                # 기준 : currentPos[1][1]
                operator = 1 if (center[1] - self.currentPos[1][1] > 0) else -1
                self.currentPos[1][1] = self.currentPos[1][1] + operator
                
    def diverge(self):
        # center : 두 점의 중간 점
        # operator : +1, -1 둘 중 하나를 가지며 픽셀 이동을 위한 값
        # length : 두 점 사이의 거리
        
        self.move += MOVE
        
        center = [int((self.currentPos[0][0] + self.currentPos[1][0]) / 2), int((self.currentPos[0][1] + self.currentPos[1][1]) / 2)]
        operator = 1
        
        # 일직선 상(축)에 위치한 경우
        if (self.currentPos[0][0] == self.currentPos[1][0]):
            if (self.currentPos[0][1] < self.currentPos[1][1] and self.currentPos[0][1] > 2 and self.currentPos[1][1] < 398):
                self.currentPos[0][1] = self.currentPos[0][1] - 1
                self.currentPos[1][1] = self.currentPos[1][1] + 1
                
            elif(self.currentPos[0][1] > self.currentPos[1][1] and self.currentPos[0][1] < 398 and self.currentPos[1][1] > 2) :
                self.currentPos[0][1] = self.currentPos[0][1] + 1
                self.currentPos[1][1] = self.currentPos[1][1] - 1
                
        elif (self.currentPos[0][1] == self.currentPos[1][1]):
            if (self.currentPos[0][0] < self.currentPos[1][0] and self.currentPos[0][0] > 2 and self.currentPos[1][0] < 398):
                self.currentPos[0][0] = self.currentPos[0][0] - 1
                self.currentPos[1][0] = self.currentPos[1][0] + 1
                
            elif (self.currentPos[0][0] > self.currentPos[1][0] and self.currentPos[0][0] < 398 and self.currentPos[1][0] > 2):
                self.currentPos[0][0] = self.currentPos[0][0] + 1
                self.currentPos[1][0] = self.currentPos[1][0] - 1

        # 일직선 상(축)에 위치하지 않은 경우
        else :
            if (2 < self.currentPos[0][0] and self.currentPos[0][0] < 398 and 2 < self.currentPos[0][1] and self.currentPos[0][1] < 398):
                if (2 < self.currentPos[1][0] and self.currentPos[1][0] < 398 and 2 < self.currentPos[1][1] and self.currentPos[1][1] < 398):
                    # 기준 : currentPos[0][0]
                    operator = -1 if (center[0] - self.currentPos[0][0] > 0) else 1
                    self.currentPos[0][0] = self.currentPos[0][0] + operator
                                            
                    # 기준 : currentPos[0][1]
                    operator = -1 if (center[1] - self.currentPos[0][1] > 0) else 1
                    self.currentPos[0][1] = self.currentPos[0][1] + operator
                    
                    # 기준 : currentPos[1][0]
                    operator = -1 if (center[0] - self.currentPos[1][0] > 0) else 1
                    self.currentPos[1][0] = self.currentPos[1][0] + operator
                    
                    # 기준 : currentPos[1][1]
                    operator = -1 if (center[1] - self.currentPos[1][1] > 0) else 1
                    self.currentPos[1][1] = self.currentPos[1][1] + operator

    def observe(self):
        posMap = np.zeros([400, 400])
        posMap[self.currentPos[0][0]][self.currentPos[0][1]] = 255
        posMap[self.currentPos[1][0]][self.currentPos[1][1]] = 255
        
        return [x + y for x, y in zip(self.currentMap, posMap)]
    
    def score(self):
        score = 0.0
        done = False
        isContained = None
        up, down = 0, 0
        between = False
        
        # 두 점의 물체 포함 유무
        for col, row in (self.coorList):
            if self.currentPos[0][0] == col and self.currentPos[0][1] == row or self.currentPos[1][0] == col and self.currentPos[1][1] == row:
                isContained = True

        if isContained:
            score -= 100
        
        # 두 점의 거리와 그리퍼 크기 비교
        length = math.sqrt((self.currentPos[0][0] - self.currentPos[1][0])**2 + (self.currentPos[0][1] - self.currentPos[1][1])**2)
        
        if (length > 250) :
            score -= 100
        
        # 넓이 양분, 두 점 사이에 물체가 있는지 확인
        if (self.currentPos[0][0] - self.currentPos[1][0] == 0 ) :
            m = 9999
        else :
            m = (self.currentPos[0][1] - self.currentPos[1][1]) / (self.currentPos[0][0] - self.currentPos[1][0])
        
        c = self.currentPos[0][1] - m * self.currentPos[0][0]
        
        for coor in self.coorList :
            if ((m * coor[0]) + c) > coor[1] :
                up += 1
            elif ((m * coor[0]) + c) < coor[1] :
                down += 1
            else :
                between = True

        if (between):
            score = (160000 - abs(up - down))/400
        else:
            score -= 200
            done = False

        # 액션의 갯수를 점수에 반영
        score -= self.move

        if score > self.max:
            self.max = score
            done = True
        else:
            done = False
            
        # 횟수 제한
        if (self.move == 100):
            done = True

        return score, done