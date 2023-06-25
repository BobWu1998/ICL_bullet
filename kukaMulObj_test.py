#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

# from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
import time

from typing import Optional
from robotics_transformer.film_efficientnet import pretrained_efficientnet_encoder
from robotics_transformer.tokenizers import token_learner
import tensorflow as tf

from env.kuka_env import Kuka_RT1
from options import KukaOptions

class KukaTest():
  def __init__(self) -> None:
    options = KukaOptions()
    self.args = options.parse_args()
    self.environment = Kuka_RT1(renders = True, 
                                        isDiscrete = False, 
                                        maxSteps = 10000000, 
                                        width = self.args.img_res[0], 
                                        height = self.args.img_res[1])

  def add_motors(self):
    self.motorsIds = []

    dv = 0.01
    self.motorsIds.append(self.environment._p.addUserDebugParameter("posX", -dv, dv, 0))
    self.motorsIds.append(self.environment._p.addUserDebugParameter("posY", -dv, dv, 0))
    self.motorsIds.append(self.environment._p.addUserDebugParameter("posZ", -dv, dv, 0))
    self.motorsIds.append(self.environment._p.addUserDebugParameter("yaw", -dv, dv, 0))
    self.motorsIds.append(self.environment._p.addUserDebugParameter("fingerAngle", 0, 0.3, .3))

  def main(self):
    self.environment.reset()
    self.add_motors()

    done = False
    while (not done):

      action = []
      for motorId in self.motorsIds:
        action.append(self.environment._p.readUserDebugParameter(motorId))
      obs, reward, done, info = self.environment.step(action)
      print(obs.shape)
      
if __name__ == "__main__":
  while (1):
    env = KukaTest()
    env.main()
    env = None