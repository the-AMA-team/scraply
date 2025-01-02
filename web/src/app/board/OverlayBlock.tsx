"use client";
import React from "react";
import { ActivationFunction, UILayer } from "~/types";

interface OverlayBlockProps {
  id: string;
  label: string;
  color: string;
  block: UILayer;
  setCanvasBlocks: React.Dispatch<React.SetStateAction<UILayer[]>>;
}

const OverlayBlock = ({
  id,
  label,
  color,
  block,
  setCanvasBlocks,
}: OverlayBlockProps) => {
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
          setCanvasBlocks((prevBlocks) =>
            prevBlocks.map((block) =>
              block.id === id ? { ...block, neurons: newNeurons } : block,
            ),
          );
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
                setCanvasBlocks((prevBlocks) =>
                  prevBlocks.map((block) =>
                    block.id === id
                      ? {
                          ...block,
                          activationFunction:
                            newActivationFunction as ActivationFunction,
                        }
                      : block,
                  ),
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
