"use client";
import React from "react";
import { Config, UILayer, hasActivationFunction } from "~/types/index";
import { configToBlocks } from "~/util/configToBlocks";
import {
  BlockNeurons,
  BlockParams,
  BlockActivation,
} from "~/util/blockRenderUtils";

interface ModelMiniMapProps {
  trainingConfig: Config;
}

const ModelMiniMap: React.FC<ModelMiniMapProps> = ({ trainingConfig }) => {
  const blocks = configToBlocks(trainingConfig);

  return (
    <div className="flex flex-col gap-2">
      {blocks.map((block) => (
        <MiniBlock key={block.id} block={block} />
      ))}
    </div>
  );
};

interface MiniBlockProps {
  block: UILayer;
}

const MiniBlock: React.FC<MiniBlockProps> = ({ block }) => {
  return (
    <div
      className="flex items-center gap-2 rounded-lg border-l-4 p-3 text-white shadow-md transition-all duration-100 hover:brightness-110"
      style={{
        backgroundColor: `${block.color}`,
        borderLeftColor: block.color,
        boxShadow: `0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06), 2px 0 8px -2px ${block.color}30`,
      }}
    >
      {/* Layer name */}
      <div className="flex-shrink-0">
        <h4 className="text-sm font-semibold text-white">{block.label}</h4>
      </div>

      {/* Neurons */}
      <BlockNeurons block={block} />

      {/* Parameters */}
      <BlockParams block={block} />

      {/* Activation function */}
      {hasActivationFunction(block) && block.activationFunction !== "ReLU" && (
        <div className="ml-auto flex-shrink-0">
          <BlockActivation block={block} />
        </div>
      )}
    </div>
  );
};

export default ModelMiniMap;
