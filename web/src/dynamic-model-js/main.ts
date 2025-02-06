// TRAINING FUNCTIONS FOR NON-DICTIONARY PLAN
import * as tf from "@tensorflow/tfjs";
import { activations, layers, losses, optimizers } from "./main-params";

// TEMP data
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
console.log("input: " + xs.toString());
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);
console.log("output: " + ys.toString());

// TEMP model declaration
const model = tf.sequential();
model.add(layers["Linear"]({ units: 1, inputShape: [1] }));

// compiling model stuff
model.compile({ loss: losses["MSE"], optimizer: optimizers["Adam"], metrics: ['accuracy']}); // NEED TO USE METRICS ARG TO STORE ACCURACY

// training stuff
const h = await model.fit(xs, ys, { epochs: 10, batchSize: 1 });  
// need await because javascript needs to wait for it to finish before moving on 😛
// could use early stopping as a feature for the user??
// TRAINING LOG!!
console.log(h.history.loss); // 10 loss values
console.log(h.history.acc); // 10 acc values

// testing stuff
const result = model.evaluate(tf.tensor2d([1, 2, 3, 4], [4, 1]), tf.tensor2d([1, 3, 5, 7], [4, 1]), {batchSize: 4});
//const result1 = model.evaluate(tf.tensor2d([1, 2, 3, 4], [4, 1]), tf.tensor2d([1, 3, 5, 7], [4, 1]));

// // Print the loss (method 1 of retrival)
// const loss = (result[0] as tf.Scalar).dataSync()[0];
// const accuracy = (result[1] as tf.Scalar).dataSync()[0];
// console.log(`individual Loss: ${loss}`); // first one is loss, second one is accuracy
// console.log(`individual Accuracy: ${accuracy}`); // first one is loss, second one is accuracy

// Print the loss (method 2 of retrival)
const loss = (result instanceof Array ? result[0] : result)?.dataSync();
const accuracy = (result instanceof Array ? result[1] : result)?.dataSync();
console.log(`slay Loss: ${loss}`); // first one is loss, second one is accuracy
console.log(`slay Accuracy: ${accuracy}`); // first one is loss, second one is accuracy


// prediction/inference stuff
const yhaty = model.predict(tf.tensor2d([1, 2, 3, 4], [4, 1]));
console.log(yhaty.toString());
const yhat = model.predict(tf.tensor2d([5], [1, 1]));
console.log(yhat.toString());
console.log("Done training! 👾")
