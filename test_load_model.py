import tensorflow as tf
import tensorflow_probability as tfp  # This line is important
print(tf.__version__)
_dir = '/home/bob/UQ/robotics_transformer/trained_checkpoints/rt1multirobot'

# # If you have the saved_model.pb file in a directory, the directory path should be given
model = tf.saved_model.load(_dir)

# Alternatively, you can use the following if the model was saved using Keras
# model = tf.keras.models.load_model('path_to_your_model_directory')

# imported = tf.saved_model.load(_dir)
# assert imported(tf.constant(3.)).numpy() == 3
# imported.mutate(tf.constant(2.))
# assert imported(tf.constant(3.)).numpy() == 6
# reader = tf.train.load_checkpoint(_dir)
# shape_from_key = reader.get_variable_to_shape_map()
# dtype_from_key = reader.get_variable_to_dtype_map()

# sorted(shape_from_key.keys())


saved_path = '/home/bob/playground/robotics_transformer/trained_checkpoints/rt1main'
from tf_agents.policies import py_tf_eager_policy

py_tf_eager_policy.SavedModelPyTFEagerPolicy(
    model_path=saved_path,
    load_specs_from_pbtxt=True,
    use_tf_function=True,
)