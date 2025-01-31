import * as tf from "@tensorflow/tfjs";

// still need to add function for are_params_valid  AND add dataloaders

// export const dataloaders = {
// will need to update this with new dataloaders
// }

// use tf.data.csv(source, csvConfig); for PIMA dataset --> using CSV for now

export const activations = {
  ReLU: tf.relu,
  Sigmoid: tf.sigmoid,
  Tanh: tf.tanh,
  Softmax: tf.softmax,
  LeakyReLU: tf.leakyRelu,
  PReLU: tf.prelu,
};

export const layers = {
  Flatten: tf.layers.flatten,
  Linear: tf.layers.dense,
  Conv1D: tf.layers.conv1d,
  Conv2D: tf.layers.conv2d,
  Conv3D: tf.layers.conv3d,
  LSTM: tf.layers.lstm,
  GRU: tf.layers.gru,
  RNN: tf.layers.rnn,
  Dropout: tf.layers.dropout,
};
// implementation example
// layers["Dense"]({units: 32, inputShape: [50]})

export const losses = {
  BCE: tf.losses.sigmoidCrossEntropy,
  CrossEntropy: tf.losses.softmaxCrossEntropy,
  MSE: tf.losses.meanSquaredError,
};

// used keras documentation as a guide for default lr
export const optimizers = {
  Adam: tf.train.adam(0.001),
  SGD: tf.train.sgd(0.01),
  RMSprop: tf.train.rmsprop(0.001),
};
