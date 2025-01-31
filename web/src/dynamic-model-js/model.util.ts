import * as tf from "@tensorflow/tfjs";

export const createTfModel = (layers: tf.layers.Layer[]) => {
  const model = tf.sequential();
  console.log("utils", layers);
  layers.forEach((layer) => {
    model.add(layer);
  });
  console.log(model);
  return model;
};

export const trainTfModel = async (
  model: tf.Sequential,
  xs: tf.Tensor,
  ys: tf.Tensor,
  epochs: number,
) => {
  model.compile({ loss: "meanSquaredError", optimizer: "adam" });
  await model.fit(xs, ys, { epochs });
};
