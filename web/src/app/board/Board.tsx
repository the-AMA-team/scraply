"use client";
import React, { useState } from "react";
import LayersBoard from "./LayersBoard";
import { AppMode } from "~/types";
import Toggle from "../_components/Toggle";
import TransformersBoard from "./TransformersBoard";

const Board = () => {
  const [mode, setMode] = useState<AppMode>(AppMode.LAYERS);

  // global layer state
  const lossState = useState("BCE");
  const optimizerState = useState("Adam");
  const learningRateState = useState(0.001);
  const epochState = useState(100);
  const batchSizeState = useState(10);

  const isTrainingState = useState(false);
  const trainingResState = useState<any | null>(null);
  const progressState = useState(0);

  const isLoadingSuggestionsState = useState(false);

  return (
    <div
      className={`${mode === AppMode.TRANSFORMERS ? "h-full min-h-screen" : "h-screen overflow-hidden"} bg-zinc-900 text-white`}
    >
      <div>
        <div className="flex justify-center p-4">
          <Toggle
            color="blue"
            option2="TRANSFORMERS"
            option1="LAYERS"
            selected={mode}
            setSelected={
              setMode as React.Dispatch<React.SetStateAction<string>>
            }
          />
        </div>
        {mode === AppMode.TRANSFORMERS ? (
          <TransformersBoard />
        ) : (
          <LayersBoard
            lossState={lossState}
            optimizerState={optimizerState}
            learningRateState={learningRateState}
            epochState={epochState}
            batchSizeState={batchSizeState}
            isTrainingState={isTrainingState}
            trainingResState={trainingResState}
            progressState={progressState}
            isLoadingSuggestionsState={isLoadingSuggestionsState}
          />
        )}
      </div>
    </div>
  );
};

export default Board;
