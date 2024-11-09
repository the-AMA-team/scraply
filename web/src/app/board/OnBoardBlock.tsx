"use client";
import { useArchitecture } from "@/contexts/ArchitectureContext";
import { Block } from "@/types";
import React, { useEffect, useState } from "react";

interface OnBoardBlockProps {
  id: string;
  label: string;
  color: string;
}

const OnBoardBlock = ({ id, label, color }: OnBoardBlockProps) => {
  const { canvasBlocks, setCanvasBlocks } = useArchitecture()!;
  return (
    <div
      className={`px-8 py-20 rounded-2xl text-center ring-1 ring-zinc-100 mr-1`}
      style={{ backgroundColor: color }}
    >
      <div className="text-xl font-medium">{label}</div>
      <div className="relative overflow-visible flex">
        <div>
          <select
            className="h-16 w-16 rounded-full text-zinc-900 outline-none absolute right-[-65px] top-[-20px] text-sm"
            value={
              canvasBlocks.find((block) => block.id === id)
                ?.activationFunction as string
            }
            onChange={(e) => {
              const newActivationFunction = e.target.value;
              setCanvasBlocks((prevBlocks) =>
                prevBlocks.map((block) =>
                  block.id === id
                    ? { ...block, activationFunction: newActivationFunction }
                    : block
                )
              );
              console.log(setCanvasBlocks);
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
    </div>
  );
};

export default OnBoardBlock;
