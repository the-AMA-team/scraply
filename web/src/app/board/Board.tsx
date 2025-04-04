"use client";
import React, { useEffect, useState } from "react";
import LayersBoard from "./LayersBoard";
import { AppMode, Dataset } from "~/types";
import Toggle from "../_components/Toggle";
import TransformersBoard from "./TransformersBoard";
import DATASETS from "~/util/DATASETS";

const Board = () => {
  const [mode, setMode] = useState<AppMode>(AppMode.LAYERS);
  const [datasetOptions, setDatasetOptions] = useState<Dataset[]>(
    DATASETS[mode],
  );
  const [selectedDataset, setSelectedDataset] = useState<string>(
    datasetOptions[0]!.inputName,
  );

  useEffect(() => {
    setDatasetOptions(DATASETS[mode]);
    setSelectedDataset(DATASETS[mode][0]!.inputName);
  }, [mode]);

  // global layer state
  const trainingBlockToggleState = useState(true);
  const lossState = useState("BCE");
  const optimizerState = useState("Adam");
  const learningRateState = useState(0.001);
  const epochState = useState(100);
  const batchSizeState = useState(10);

  const isTrainingState = useState(false);
  const trainingResHistoryState = useState<any[]>([]);
  const progressState = useState(0);

  const isLoadingSuggestionsState = useState(false);

  const resultsBlockToggleState = useState(true);

  const [permission, setPermission] = useState("denied");

  useEffect(() => {
    if ("Notification" in window) {
      Notification.requestPermission().then(setPermission);
    }
    console.log(permission);
  }, []);

  const showNotification = (title: string, body: string) => {
    if (!("Notification" in window)) return;
    if (permission === "granted") {
      new Notification(title, {
        body,
        // icon: "favicon.png",
      });
    }
  };

  return (
    <div
      className={`${mode === AppMode.TRANSFORMERS ? "" : "overflow-hidden"} bg-zinc-900 text-white`}
    >
      <div>
        <div className="flex justify-between p-4">
          <div className="mx-4 flex items-center">
            <div className="mx-2 text-lg">Dataset</div>
            <select
              className="rounded bg-zinc-800 p-2 text-white outline-none"
              onChange={(e) => setSelectedDataset(e.target.value)}
              value={selectedDataset}
            >
              {datasetOptions.map((dataset, idx) => {
                return (
                  <option
                    key={idx}
                    value={dataset.inputName}
                    className="text-wrap"
                  >
                    {dataset.label}
                  </option>
                );
              })}
            </select>
          </div>
          <Toggle
            color="blue"
            option2="TRANSFORMERS"
            option1="LAYERS"
            selected={mode}
            setSelected={
              setMode as React.Dispatch<React.SetStateAction<string>>
            }
          />
          <div className="invisible mx-4 flex items-center">
            <div className="mx-2 text-xl">Dataset</div>
            <select
              className="rounded bg-zinc-800 p-2 text-white outline-none"
              onChange={(e) => console.log(e.target.value)}
              value={selectedDataset}
            >
              {datasetOptions.map((dataset, idx) => {
                return (
                  <option key={idx} value={dataset.inputName}>
                    {dataset.label}
                  </option>
                );
              })}
            </select>
          </div>
        </div>
        {mode === AppMode.TRANSFORMERS ? (
          <TransformersBoard selectedDataset={selectedDataset} />
        ) : (
          <LayersBoard
            selectedDataset={selectedDataset}
            trainingBlockToggleState={trainingBlockToggleState}
            lossState={lossState}
            optimizerState={optimizerState}
            learningRateState={learningRateState}
            epochState={epochState}
            batchSizeState={batchSizeState}
            isTrainingState={isTrainingState}
            trainingResHistoryState={trainingResHistoryState}
            progressState={progressState}
            isLoadingSuggestionsState={isLoadingSuggestionsState}
            resultsBlockToggleState={resultsBlockToggleState}
            showNotification={showNotification}
          />
        )}
      </div>
    </div>
  );
};

export default Board;
