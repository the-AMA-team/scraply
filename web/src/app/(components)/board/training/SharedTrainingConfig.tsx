"use client";
import React, { useEffect, useState } from "react";
import { LossFunction, OptimizerType } from "~/types/index";
import {
  TRAINING_DEFAULTS,
  LOSS_FUNCTIONS,
  OPTIMIZERS,
  getMaxEpochsForDataset,
  isImageDataset,
} from "~/util/trainingConfig";
import TrainingConfigItem from "./TrainingConfigItem";

interface SharedTrainingConfigProps {
  // Configuration values
  loss: LossFunction;
  optimizer: OptimizerType;
  learningRate: number;
  epochs: number;
  batchSize: number;
  runName: string;
  selectedDataset?: string;

  // Setters
  setLoss: (loss: LossFunction) => void;
  setOptimizer: (optimizer: OptimizerType) => void;
  setLearningRate: (rate: number) => void;
  setEpochs: (epochs: number) => void;
  setBatchSize: (size: number) => void;
  setRunName: (name: string) => void;

  // Optional reset functions
  onResetLoss?: () => void;
  onResetOptimizer?: () => void;
  onResetLearningRate?: () => void;
  onResetEpochs?: () => void;
  onResetBatchSize?: () => void;
  onResetRunName?: () => void;
}

const SharedTrainingConfig: React.FC<SharedTrainingConfigProps> = ({
  loss,
  optimizer,
  learningRate,
  epochs,
  batchSize,
  runName,
  selectedDataset,
  setLoss,
  setOptimizer,
  setLearningRate,
  setEpochs,
  setBatchSize,
  setRunName,
  onResetLoss,
  onResetOptimizer,
  onResetLearningRate,
  onResetEpochs,
  onResetBatchSize,
  onResetRunName,
}) => {
  const maxEpochs = selectedDataset
    ? getMaxEpochsForDataset(selectedDataset)
    : TRAINING_DEFAULTS.epochs.max;
  const [epochsError, setEpochsError] = useState<string | null>(null);

  useEffect(() => {
    setEpochsError(null);
  }, [selectedDataset]);

  const handleEpochsChange = (value: number) => {
    if (Number.isNaN(value)) return;
    if (value > maxEpochs) {
      setEpochsError(
        selectedDataset && isImageDataset(selectedDataset)
          ? `Max ${maxEpochs} epochs allowed for image datasets`
          : `Max ${maxEpochs} epochs allowed`,
      );
      setEpochs(maxEpochs);
      return;
    }
    setEpochsError(null);
    setEpochs(Math.max(value, TRAINING_DEFAULTS.epochs.min));
  };

  return (
    <div className="space-y-3">
      <TrainingConfigItem title="Run Name" onReset={onResetRunName}>
        <input
          type="text"
          value={runName}
          onChange={(e) => setRunName(e.target.value)}
          placeholder="(Optional) Name training run"
          className="w-full rounded-lg bg-zinc-700 px-3 py-1.5 text-base text-zinc-100 placeholder-zinc-400 outline-none focus:bg-zinc-600"
        />
      </TrainingConfigItem>

      <div className="text-xl text-zinc-700">Model</div>

      <TrainingConfigItem title="Loss Function" onReset={onResetLoss}>
        <select
          value={loss}
          onChange={(e) => setLoss(e.target.value as LossFunction)}
          className="rounded-lg bg-zinc-700 px-3 py-1.5 text-base outline-none"
        >
          {LOSS_FUNCTIONS.map((fn) => (
            <option key={fn.value} value={fn.value}>
              {fn.label}
            </option>
          ))}
        </select>
      </TrainingConfigItem>

      <TrainingConfigItem title="Optimizer" onReset={onResetOptimizer}>
        <select
          value={optimizer}
          onChange={(e) => setOptimizer(e.target.value as OptimizerType)}
          className="rounded-lg bg-zinc-700 px-3 py-1.5 text-base outline-none"
        >
          {OPTIMIZERS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </TrainingConfigItem>

      <div className="mt-3 text-xl text-zinc-700">Training</div>

      <TrainingConfigItem title="Learning Rate" onReset={onResetLearningRate}>
        <div className="flex items-center">
          <input
            type="range"
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            min={TRAINING_DEFAULTS.learningRate.min}
            max={TRAINING_DEFAULTS.learningRate.max}
            step={TRAINING_DEFAULTS.learningRate.step}
            className="flex-1"
            name="learningRate"
          />
          <input
            type="number"
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            min={TRAINING_DEFAULTS.learningRate.min}
            max={TRAINING_DEFAULTS.learningRate.max}
            step={TRAINING_DEFAULTS.learningRate.step}
            className="mx-2 w-20 rounded-lg bg-zinc-700 px-2 py-1 text-center text-base outline-none"
            name="learningRate"
          />
        </div>
      </TrainingConfigItem>

      <TrainingConfigItem title="Epochs" onReset={onResetEpochs}>
        <div className="flex flex-col items-end">
          <div className="flex items-center">
            <input
              type="range"
              value={Math.min(epochs, maxEpochs)}
              onChange={(e) => handleEpochsChange(parseInt(e.target.value))}
              min={TRAINING_DEFAULTS.epochs.min}
              max={maxEpochs}
              className="flex-1"
              name="epochs"
            />
            <input
              type="number"
              value={epochs}
              onChange={(e) => handleEpochsChange(parseInt(e.target.value))}
              min={TRAINING_DEFAULTS.epochs.min}
              max={maxEpochs}
              className={`mx-2 w-20 rounded-lg bg-zinc-700 px-2 py-1 text-center text-base outline-none ${
                epochsError ? "ring-1 ring-red-500" : ""
              }`}
              name="epochs"
            />
          </div>
          {epochsError && (
            <p className="mt-1 max-w-[220px] text-right text-xs text-red-400">
              {epochsError}
            </p>
          )}
        </div>
      </TrainingConfigItem>

      <TrainingConfigItem title="Batch Size" onReset={onResetBatchSize}>
        <div className="flex items-center">
          <input
            type="range"
            value={batchSize}
            onChange={(e) => setBatchSize(parseInt(e.target.value))}
            min={TRAINING_DEFAULTS.batchSize.min}
            max={TRAINING_DEFAULTS.batchSize.max}
            className="flex-1"
            name="batchSize"
          />
          <input
            type="number"
            value={batchSize}
            onChange={(e) => setBatchSize(parseInt(e.target.value))}
            min={TRAINING_DEFAULTS.batchSize.min}
            max={TRAINING_DEFAULTS.batchSize.max}
            className="mx-2 w-20 rounded-lg bg-zinc-700 px-2 py-1 text-center text-base outline-none"
            name="batchSize"
          />
        </div>
      </TrainingConfigItem>
    </div>
  );
};

export default SharedTrainingConfig;
