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

const OverlayBlock = ({ id, label, color, block }: OverlayBlockProps) => {
  const { changeActivationFunction, changeNeurons } = useBoardStore();
  return (
    <div
      className={`mr-1 cursor-grab rounded-2xl px-10 py-20 text-center ring-1 ring-zinc-100`}
      style={{ backgroundColor: color }}
    >
      <input
        className="h-8 w-10 rounded-md text-center text-zinc-900 outline-none"
        type="number"
        value={block?.neurons}
        onChange={(e) => {
          const newNeurons = parseInt(e.target.value);
          if (newNeurons < 1) return;
          changeNeurons(id, newNeurons);
        }}
      />
      <div className="text-xl font-medium">{label}</div>
      {block?.activationFunction && (
        <div className="relative flex overflow-visible">
          <div>
            <select
              className="absolute right-[-75px] top-[-20px] h-16 w-16 cursor-pointer rounded-full bg-zinc-100 text-sm text-zinc-900 outline-none"
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
        </div>
      )}
    </div>
  );
};

export default OverlayBlock;
