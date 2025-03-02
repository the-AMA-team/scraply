import { Dataset } from "~/types";

const LAYERS: Dataset[] = [
  {
    label: "Pima Diabetes",
    inputName: "pima",
  },
];

const TRANSFORMERS: Dataset[] = [
  {
    label: "Alice in Wonderland",
    inputName: "alice",
  },
  {
    label: "Shakespeare",
    inputName: "shakespeare",
  },
];

const DATASETS = {
  LAYERS,
  TRANSFORMERS,
};

export default DATASETS;
