import * as tf from '@tensorflow/tfjs';
import { activations, layers, losses, optimizers } from './main-params';

// testing main-params dictionaries

// request user input here?? 

// temp data
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// model declaration
const model = tf.sequential();
model.add(layers["Dense"]({units: 1, inputShape: [1]}));
model.compile({loss: losses["MSE"], optimizer: optimizers["Adam"]});


// Train the model using the data.
model.fit(xs, ys, {epochs: 10}).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  const slay = model.predict(tf.tensor2d([5], [1, 1]));
  console.log(slay.toString());

  console.log('Done training');
  // Open the browser devtools to see the output

});