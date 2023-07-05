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

import tensorflow as tf
import tensorflow_probability as tfp  # This line is important
import tensorflow_hub as hub

from tf_agents.policies import py_tf_eager_policy
import numpy as np
from tf_agents.trajectories import TimeStep

class TfAgentTest():
    def __init__(self) -> None:
        options = KukaOptions()
        self.args = options.parse_args()
        self.args = options.parse_args()
        # self.environment = Kuka_RT1(renders = True, 
        #                                     isDiscrete = False, 
        #                                     maxSteps = 10000000, 
        #                                     width = self.args.img_res[0], 
        #                                     height = self.args.img_res[1])
        
        # load RT-1 and universal sentence embedding
        # self.model = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        #                 model_path=self.args.model_dir,
        #                 load_specs_from_pbtxt=True,
        #                 use_tf_function=True,
        #                 )
        self.model = tf.saved_model.load(self.args.model_dir)
        self.use = hub.load(self.args.USE_dir) # universal sentence encoder
    
    def get_fake_data_input(self):
        # Define the batch size and sequence length
        batch_size = 1
        sequence_length = 6

        # For TimeStep
        step_type = np.random.randint(0, 2, size=(batch_size)).astype(np.int32)
        reward = np.random.uniform(0, 1, size=(batch_size)).astype(np.float32)
        discount = np.full((batch_size), 0.99).astype(np.float32)

        # For observation
        observation = {
            'gripper_closed': np.random.rand(batch_size, 1).astype(np.float32),
            'workspace_bounds': np.random.rand(batch_size, 3, 3).astype(np.float32),
            'natural_language_embedding': np.random.rand(batch_size, 512).astype(np.float32),
            'orientation_start': np.random.rand(batch_size, 4).astype(np.float32),
            'image': np.random.randint(0, 256, size=(batch_size, 256, 320, 3)).astype(np.uint8),
            'base_pose_tool_reached': np.random.rand(batch_size, 7).astype(np.float32),
            'height_to_bottom': np.random.rand(batch_size, 1).astype(np.float32),
            'orientation_box': np.random.rand(batch_size, 2, 3).astype(np.float32),
            'natural_language_instruction': np.array([b'This is a fake instruction']*batch_size, dtype=np.object),
            'vector_to_go': np.random.rand(batch_size, 3).astype(np.float32),
            'src_rotation': np.random.rand(batch_size, 4).astype(np.float32),
            'gripper_closedness_commanded': np.random.rand(batch_size, 1).astype(np.float32),
            'robot_orientation_positions_box': np.random.rand(batch_size, 3, 3).astype(np.float32),
            'rotation_delta_to_go': np.random.rand(batch_size, 3).astype(np.float32)
        }

        time_step = {
            'step_type': step_type,
            'reward': reward,
            'discount': discount,
            'observation': observation
        }
        time_step_data = TimeStep(
            step_type=time_step['step_type'],
            reward=time_step['reward'],
            discount=time_step['discount'],
            observation=time_step['observation'],
        )

        # For ActionTokens, etc
        action_tokens = np.random.randint(0, 2, size=(batch_size, sequence_length, 11, 1, 1)).astype(np.int32)
        image_seq = np.random.randint(0, 256, size=(batch_size, sequence_length, 256, 320, 3)).astype(np.uint8)
        step_num = np.random.randint(0, 10, size=(batch_size, 1, 1, 1, 1)).astype(np.int32)
        t = np.random.randint(0, 10, size=(batch_size, 1, 1, 1, 1)).astype(np.int32)

        aux_input = {
            'action_tokens': action_tokens,
            'image': image_seq,
            'step_num': step_num,
            't': t
        }

        # Model input
        model_input = (time_step_data, aux_input, None)
        return model_input
    
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

    instruction = ['pick up an object']

    input_data = {'image': None, 
                  'natural_language_embedding': self.use(instruction)}

    done = False
    while (not done):

      action = []
      for motorId in self.motorsIds:
        action.append(self.environment._p.readUserDebugParameter(motorId))
      obs, reward, done, info = self.environment.step(action)

      input_data['image'] = tf.expand_dims(obs, axis=0)
      # print(obs)
      output_tokens = self.model(input_data)
      

      print(input_data['image'])


if __name__ == "__main__":
    env = TfAgentTest()
    print(dir(env.model))
    # # if load the data with tf_agent, use the following to check the wanted input format
    # print(env.model.time_step_spec)
    # print(env.model.action_spec)

    # Get the input tensor information
    fake_input = env.get_fake_data_input()

    res = env.model.action(fake_input[0], fake_input[1], None)
    print(res)











