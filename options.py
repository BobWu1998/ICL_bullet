import argparse

class KukaOptions():
  def __init__(self) -> None:
    self.parser = argparse.ArgumentParser()
    self.parser.add_argument('--img_res', default=[256, 256])
    self.parser.add_argument('--task', default='state_feedback_imitation_learning')
    self.parser.add_argument('--model_name', default='s4')

  def parse_args(self):
    self.args = self.parser.parse_args()
    return self.args
