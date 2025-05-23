"use client";
import React, { useState } from "react";
import Toggle from "../_components/Toggle";
import Layers from "./Layers";
import Data from "./Data";
import { AppMode, DataMode } from "~/types";

const Board = () => {
  const [mode, setMode] = useState<AppMode>(AppMode.DATA);

  // global data state
  const dataModeState = useState<DataMode>(DataMode.UPLOAD);

  // global layer state
  const lossState = useState("BCE");
  const optimizerState = useState("Adam");
  const learningRateState = useState(0.001);
  const epochState = useState(100);
  const batchSizeState = useState(10);

  const isTrainingState = useState(false);
  const trainingResState = useState<any | null>(null);
  const progressState = useState(0);

  return (
    <div>
      <div className="flex justify-center p-4">
        <Toggle
          color="blue"
          option1="DATA"
          option2="LAYERS"
          selected={mode}
          setSelected={setMode as React.Dispatch<React.SetStateAction<string>>}
        />
      </div>
      {mode === "DATA" ? (
        <Data dataModeState={dataModeState} />
      ) : (
        <Layers
          lossState={lossState}
          optimizerState={optimizerState}
          learningRateState={learningRateState}
          epochState={epochState}
          batchSizeState={batchSizeState}
          isTrainingState={isTrainingState}
          trainingResState={trainingResState}
          progressState={progressState}
        />
      )}
    </div>
  );
};

export default Board;
