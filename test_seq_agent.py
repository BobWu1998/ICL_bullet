# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for sequence_agent."""
from robotics_transformer.sequence_agent_test_set_up import SequenceAgentTestSetUp, DummyActorNet
import tensorflow as tf
from tf_agents.agents import data_converter
from options import KukaOptions

class SequenceAgentTest(SequenceAgentTestSetUp):

  def testAsTransitionType(self):
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
    # my_actor_net = DummyActorNet()
    # my_actor_net.set_weights(self.model.get_weights())
    # for var, value in self.model.model_variables.items():
    #   my_actor_net[var].assign(value)
    # for var_loaded, var_dummy in zip(self.model.model_variables, my_actor_net.trainable_weights):
    #   var_dummy.assign(var_loaded.value())

    agent = self.create_agent_and_initialize(actor_network=self.model.action())
    self.assertIsInstance(agent.as_transition, data_converter.AsHalfTransition)


if __name__ == '__main__':
  tf.test.main()
