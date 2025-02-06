import * as tf from "@tensorflow/tfjs";

export const createTfModel = (layers: tf.layers.Layer[]) => {
  const model = tf.sequential();
  layers.forEach((layer, idx) => {
    console.log(idx, layer);
    model.add(layer);
  });
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
