// TRAINING FUNCTIONS FOR NON-DICTIONARY PLAN

import * as tf from "@tensorflow/tfjs";
import { activations, layers, losses, optimizers } from "./main-params";

// TEMP data
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// TEMP model declaration
const model = tf.sequential();
model.add(layers["Linear"]({ units: 1, inputShape: [1] }));
model.compile({ loss: losses["MSE"], optimizer: optimizers["Adam"], metrics: ['accuracy']}); // NEED TO USE METRICS ARG TO STORE ACCURACY


const h = await model.fit(xs, ys, { epochs: 10 })  
// need await because javascript needs to wait for it to finish before moving on 😛

// could use early stopping? 
console.log(h.history.loss); // 10 loss values
console.log(h.history.acc);

const yhat = model.predict(tf.tensor2d([5], [1, 1]));
console.log(yhat.toString());
console.log("Done training! 👾")
