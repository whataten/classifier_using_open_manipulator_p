from GraspGame import GraspGame
import matplotlib.pyplot as plt
import matplotlib
import gym
from gym import spaces
import numpy as np

class GraspEnv(gym.Env):
    metadata = {'render.modes':['rgb_array']}
    
    def __init__(self):
        super(GraspEnv, self).__init__()
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.MultiDiscrete([400, 400])
        self.game = GraspGame()
        
    def step(self, action):
        
        # 상 : y가 감소하는 방향으로 두 점이 이동
        if (action == 0):
            self.game.up()
        
        # 하 : y가 증가하는 방향으로 두 점이 이동
        elif (action ==1):
            self.game.down()
        
        # 좌 : x가 감소하는 방향으로 두 점이 이동
        elif (action ==2):
            self.game.left()
        
        # 우 : x가 증가하는 방향으로 두 점이 이동
        elif (action ==3):
            self.game.right()
        
        # 반시계 : 두 점을 지나는 직선 위의 한 점 중 두 점과의 거리가 같은 점을 중심으로 반시계 방향으로 로테이션
        elif (action ==4):
            self.game.clockwise(-1)
        
        # 시계 : 두 점을 지나는 직선 위의 한 점 중 두 점과의 거리가 같은 점을 중심으로 시계 방향으로 로테이션
        elif (action ==5):
            self.game.clockwise(1)
        
        # 멀어지기 : 두 점을 지나는 직선 위의 한 점 중 두 점과의 거리가 같은 점을 중심으로 두 점이 멀어짐
        elif (action ==6):
            self.game.diverge()
        
        # 가까워지기 : # 멀어지기 : 두 점을 지나는 직선 위의 한 점 중 두 점과의 거리가 같은 점을 중심으로 두 점이 가까워짐
        elif (action ==7):
            self.game.approach()
            
        # 정지 : 두 점은 움직이지 않음
        elif (action == 8):
            pass
        
        return np.array(self.game.observe()), self.game.score()[0], self.game.score()[1], None, None
    
    def reset(self):
        self.game.reset()
        return np.array(self.game.observe())
    
    def render(self):
        state = self.game.observe()
        matplotlib.use('tkagg')
        plt.imshow(state)
        plt.show()
        
    def getState(self):
        return self.game.observe()