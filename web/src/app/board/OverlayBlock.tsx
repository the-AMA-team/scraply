"use client";
import React from "react";
import { useBoardStore } from "~/state/boardStore";
import { ActivationFunction, UILayer } from "~/types";

interface OverlayBlockProps {
  id: string;
  label: string;
  color: string;
  block: UILayer;
}

const getSpecificBlockParams = (
  id: string,
  label: string,
  changeOtherParam: (id: string, otherParam: number) => void,
  otherParam?: number,
) => {
  const KernelSize = () => {
    return (
      <div className="flex">
        <div className="m-auto">Kernel Size: </div>
        <input
          type="number"
          className="h-8 w-10 rounded-md text-center text-zinc-900 shadow-md outline-none"
          value={otherParam}
          onChange={(e) => {
            const newKernelSize = parseInt(e.target.value);
            if (newKernelSize < 1) return;
            changeOtherParam(id, newKernelSize);
          }}
        />
      </div>
    );
  };

  const HiddenSize = () => {
    return (
      <div className="flex">
        <div className="m-auto">Hidden Size: </div>
        <input
          type="number"
          className="h-8 w-10 rounded-md text-center text-zinc-900 shadow-md outline-none"
          value={otherParam}
          onChange={(e) => {
            const newHiddenSize = parseInt(e.target.value);
            if (newHiddenSize < 1) return;
            changeOtherParam(id, newHiddenSize);
          }}
        />
      </div>
    );
  };
  switch (label) {
    case "Conv1D":
      return KernelSize();
    case "Conv2D":
      return KernelSize();
    case "Conv3D":
      return KernelSize();

    case "LSTM":
      return HiddenSize();
    case "GRU":
      return HiddenSize();
    case "RNN":
      return HiddenSize();
    default:
      return null;
  }
};

const OverlayBlock = ({ id, label, color, block }: OverlayBlockProps) => {
  const { changeActivationFunction, changeNeurons, changeOtherParam } =
    useBoardStore();
  return (
    <div
      className={`cursor-grab rounded-2xl pb-3 pt-4 text-center ring-1 ring-zinc-200`}
      style={{ backgroundColor: color }}
    >
      <div className="mx-4 flex justify-between">
        <div className="text-xl font-light">{label}</div>
        <input
          className="h-8 w-10 rounded-md text-center text-zinc-900 shadow-md outline-none"
          type="number"
          value={block?.neurons}
          onChange={(e) => {
            const newNeurons = parseInt(e.target.value);
            if (newNeurons < 1) return;
            changeNeurons(id, newNeurons);
          }}
        />
      </div>
      <div className="my-2 flex justify-center text-white">
        {getSpecificBlockParams(id, label, changeOtherParam, block?.otherParam)}
      </div>
      {block?.activationFunction && (
        <div className="relative flex overflow-y-visible">
          <select
            className="absolute -bottom-8 left-1/2 -translate-x-1/2 transform cursor-pointer rounded-lg bg-zinc-100 py-2 text-center text-sm text-zinc-900 shadow-md outline-none"
            value={block?.activationFunction as string}
            onChange={(e) => {
              const newActivationFunction = e.target.value;
              changeActivationFunction(
                id,
                newActivationFunction as ActivationFunction,
              );
            }}
          >
            <option value="ReLU">ReLU</option>
            <option value="Sigmoid">Sigmoid</option>
            <option value="Tanh">Tanh</option>
            <option value="Softmax">Softmax</option>
            <option value="LeakyReLU">LeakyReLU</option>
            <option value="PReLU">PReLU</option>
          </select>
        </div>
      )}
    </div>
  );
};

export default OverlayBlock;
