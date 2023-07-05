import argparse

class KukaOptions():
  def __init__(self) -> None:
    self.parser = argparse.ArgumentParser()
    self.parser.add_argument('--img_res', default=[320, 256])
    self.parser.add_argument('--task', default='state_feedback_imitation_learning')
    self.parser.add_argument('--model_name', default='s4')
    self.parser.add_argument('--model_dir', default='/home/bob/UQ/robotics_transformer/trained_checkpoints/rt1multirobot')
    self.parser.add_argument('--USE_dir', default='/home/bob/UQ/ICL_bullet/models/USE')

  def parse_args(self):
    self.args = self.parser.parse_args()
    return self.args
