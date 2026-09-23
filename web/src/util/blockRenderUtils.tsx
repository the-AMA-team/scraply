import React from "react";
import { UILayer, hasNeurons, hasParams, hasActivationFunction } from "~/types/index";
import { PARAM_CONFIG } from "./layerConfig";

/**
 * Formats a parameter value for display
 * Converts dimension values to "1D" or "2D" format
 */
export function formatParamValue(key: string, value: number): string {
  return key === "dimension" ? `${value}D` : String(value);
}

/**
 * Gets the display configuration for a parameter
 */
export function getParamConfig(key: string) {
  return PARAM_CONFIG[key] || { shortLabel: key };
}

/**
 * Renders a parameter badge/chip component
 */
export function ParamBadge({
  paramKey,
  value,
  className = "rounded bg-white/20 px-1.5 py-0.5 font-mono text-xs text-white",
}: {
  paramKey: string;
  value: number;
  className?: string;
}) {
  const config = getParamConfig(paramKey);
  const displayValue = formatParamValue(paramKey, value);

  return (
    <span className={className} title={`${config.shortLabel}: ${displayValue}`}>
      {displayValue}
    </span>
  );
}

/**
 * Renders neuron display (input→output format)
 */
export function NeuronDisplay({
  inputNeurons,
  outputNeurons,
  className = "rounded bg-white/20 px-1.5 py-0.5 font-mono text-xs",
}: {
  inputNeurons: number;
  outputNeurons: number;
  className?: string;
}) {
  return (
    <span className={className}>
      {inputNeurons}→{outputNeurons}
    </span>
  );
}

/**
 * Renders all parameter badges for a block (excluding input/output neurons)
 */
export function BlockParams({
  block,
  className = "flex flex-wrap items-center gap-1.5",
}: {
  block: UILayer;
  className?: string;
}) {
  if (!hasParams(block)) {
    return null;
  }

  const params = block.params as Record<string, number>;
  const paramEntries = Object.entries(params).filter(
    ([key]) => key !== "inputNeurons" && key !== "outputNeurons",
  );

  if (paramEntries.length === 0) {
    return null;
  }

  return (
    <div className={className}>
      {paramEntries.map(([key, value]) => (
        <ParamBadge key={key} paramKey={key} value={value} />
      ))}
    </div>
  );
}

/**
 * Renders neuron display if the block has neurons
 */
export function BlockNeurons({
  block,
  className = "flex items-center gap-2 text-xs",
}: {
  block: UILayer;
  className?: string;
}) {
  if (!hasNeurons(block)) {
    return null;
  }

  return (
    <div className={className}>
      <NeuronDisplay
        inputNeurons={block.params.inputNeurons}
        outputNeurons={block.params.outputNeurons}
      />
    </div>
  );
}

/**
 * Renders activation function badge if present and not default (ReLU)
 */
export function BlockActivation({
  block,
  showDefault = false,
  className = "rounded bg-white/20 px-1.5 py-0.5 text-xs text-white",
}: {
  block: UILayer;
  showDefault?: boolean;
  className?: string;
}) {
  if (!hasActivationFunction(block)) {
    return null;
  }

  if (!showDefault && block.activationFunction === "ReLU") {
    return null;
  }

  return <span className={className}>{block.activationFunction}</span>;
}
