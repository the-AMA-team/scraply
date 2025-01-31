import {
  useSensors,
  useSensor,
  PointerSensor,
  DndContext,
  closestCenter,
  DragOverlay,
} from "@dnd-kit/core";
import React, { useState } from "react";
import { useBoardStore } from "~/state/boardStore";
import { startTraining, downloadFile, getConfig } from "~/util/board.util";
import { BLOCKS } from "../../util/BLOCKS";
import DraggableBlock from "./DraggableBlock";
import DroppableCanvas from "./DroppableCanvas";
import OverlayBlock from "./OverlayBlock";
import Toggle from "../_components/Toggle";
import { createTfModel } from "~/dynamic-model-js/model.util";

interface LayersProps {
  lossState: [string, React.Dispatch<React.SetStateAction<string>>];
  optimizerState: [string, React.Dispatch<React.SetStateAction<string>>];
  learningRateState: [number, React.Dispatch<React.SetStateAction<number>>];
  epochState: [number, React.Dispatch<React.SetStateAction<number>>];
  batchSizeState: [number, React.Dispatch<React.SetStateAction<number>>];
  isTrainingState: [boolean, React.Dispatch<React.SetStateAction<boolean>>];
  trainingResState: [
    any | null,
    React.Dispatch<React.SetStateAction<any | null>>,
  ];
  progressState: [number, React.Dispatch<React.SetStateAction<number>>];
}

const Layers = ({
  lossState,
  optimizerState,
  learningRateState,
  epochState,
  batchSizeState,
  isTrainingState,
  trainingResState,
  progressState,
}: LayersProps) => {
  const { canvasBlocks, activeBlock, drag } = useBoardStore();
  const [loss, setLoss] = lossState;
  const [optimizer, setOptimizer] = optimizerState;
  const [learningRate, setLearningRate] = learningRateState;
  const [epochs, setEpochs] = epochState;
  const [batchSize, setBatchSize] = batchSizeState;
  const [isTraining, setIsTraining] = isTrainingState;
  const [trainingRes, setTrainingRes] = trainingResState;
  const [progress, setProgress] = progressState;

  const [runningToggle, setRunningToggle] = useState<"TRAIN" | "HISTORY">(
    "TRAIN",
  );

  const [isModalOpen, setIsModalOpen] = useState(false);

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 5,
      },
    }),
  );

  const circleRadius = 16;
  const circumference = 2 * Math.PI * circleRadius;
  const dashArray = `${(progress / 100) * circumference} ${circumference}`;
  return (
    <DndContext
      collisionDetection={closestCenter}
      onDragStart={drag.start}
      onDragOver={drag.over}
      onDragEnd={drag.end}
      sensors={sensors}
    >
      <div
        className={`absolute left-1/2 top-1/2 z-20 h-1/2 w-1/2 -translate-x-1/2 -translate-y-1/2 transform rounded-2xl bg-zinc-800 shadow-xl ${
          !isModalOpen && "hidden"
        }`}
      >
        <div className="my-4 flex justify-center">
          <div className="w-80 rounded-full">
            <div className="relative flex flex-col items-center">
              <svg className="relative h-52 w-52" viewBox="0 0 36 36">
                <circle
                  className="text-zinc-700"
                  strokeWidth="3"
                  stroke="currentColor"
                  fill="transparent"
                  r={circleRadius}
                  cx="18"
                  cy="18"
                />
                <circle
                  className="text-green-500"
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeDasharray={dashArray}
                  stroke="currentColor"
                  fill="transparent"
                  r={circleRadius}
                  cx="18"
                  cy="18"
                  style={{ transition: "stroke-dasharray 0.3s ease" }}
                />
              </svg>

              <img
                src="/stars.png"
                alt="centered-icon"
                className="absolute left-1/2 top-1/2 w-60 -translate-x-1/2 -translate-y-1/2 transform rounded-full"
              />

              <div className="mt-2 text-center text-lg font-semibold text-green-500">
                {Math.round(progress)}% Accuracy Achieved!
              </div>
            </div>
          </div>
        </div>
        <div className="mt-4 flex justify-center">
          <button
            onClick={() => {
              setIsModalOpen(false);
            }}
            className="rounded bg-zinc-700 px-4 py-2 text-white"
          >
            YAY!
          </button>
        </div>
      </div>
      <div className={`mx-20 mt-10 flex h-full ${isModalOpen && "opacity-50"}`}>
        {/* Canvas area */}
        <div className="mr-10 flex-grow">
          <h3>Canvas</h3>
          <DroppableCanvas />
        </div>

        {/* Toolbox area */}
        <div className="mr-4">
          <h3>Scraps</h3>
          <div className="rounded-xl bg-zinc-800 py-1">
            {BLOCKS.map((block) => (
              <DraggableBlock
                key={block.id}
                id={block.id}
                label={block.label}
                color={block.color}
              />
            ))}
          </div>
        </div>

        {/* Training config */}
        <div className="">
          <h3>Train</h3>
          <div className="rounded-lg bg-zinc-800 p-1 px-2 py-1 text-sm">
            <div className="flex justify-center">
              <Toggle
                color="zinc"
                option1="TRAIN"
                option2="HISTORY"
                selected={runningToggle}
                setSelected={
                  setRunningToggle as React.Dispatch<
                    React.SetStateAction<string>
                  >
                }
              />
            </div>

            {runningToggle === "TRAIN" ? (
              <div>
                <div className="my-1 flex">
                  Loss:{" "}
                  <select
                    className="mx-1 cursor-pointer rounded bg-zinc-700 p-1 text-sm text-white outline-none"
                    value={loss}
                    onChange={(e) => setLoss(e.target.value)}
                  >
                    <option value="BCE">BCE</option>
                    <option value="CrossEntropy">CrossEntropy</option>
                  </select>
                </div>
                <div className="my-1 flex">
                  Optimizer:{" "}
                  <select
                    className="mx-1 cursor-pointer rounded bg-zinc-700 p-1 text-sm text-white outline-none"
                    value={optimizer}
                    onChange={(e) => setOptimizer(e.target.value)}
                  >
                    <option value="Adam">Adam</option>
                    <option value="AdamW">AdamW</option>
                    <option value="SGD">SGD</option>
                    <option value="RMSprop">RMSprop</option>
                  </select>
                </div>
                <div className="my-1 flex">
                  Epochs:{" "}
                  <input
                    type="range"
                    name="Batch Size"
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value))}
                    min={1}
                    max={1000}
                  />
                  <input
                    type="number"
                    className="mx-1 w-14 rounded bg-zinc-700 p-1 text-right outline-none"
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value))}
                  />
                </div>
                <div className="my-1 flex">
                  Batch Size:{" "}
                  <input
                    type="range"
                    name="Batch Size"
                    value={batchSize}
                    onChange={(e) => setBatchSize(parseInt(e.target.value))}
                    min={1}
                    max={100}
                  />
                  <input
                    type="number"
                    className="mx-1 w-14 rounded bg-zinc-700 p-1 text-right outline-none"
                    value={batchSize}
                    onChange={(e) => setBatchSize(parseInt(e.target.value))}
                  />
                </div>
                <div className="m-2 flex justify-center">
                  <button
                    disabled={isTraining}
                    className={`rounded-2xl bg-blue-500 px-4 py-2 text-lg transition-all ease-in-out ${
                      !isTraining &&
                      "hover:bg-indigo-600 hover:ring-2 active:bg-indigo-500"
                    } ring-indigo-500 duration-300 ${
                      isTraining && "animate-pulse"
                    }`}
                    // onClick={() => {
                    //   setIsTraining(true);

                    //   startTraining(
                    //     getConfig(
                    //       "pima",
                    //       canvasBlocks,
                    //       loss,
                    //       optimizer,
                    //       learningRate,
                    //       epochs,
                    //       batchSize,
                    //     ),
                    //   ).then((data: any) => {
                    //     setTrainingRes(data.RESULTS);
                    //     setProgress(
                    //       Math.round(data.RESULTS["avg_test_acc"] * 100) * 0.01,
                    //     );
                    //     setIsTraining(false);
                    //     setIsModalOpen(true);
                    //   });
                    // }}
                    onClick={() => {
                      console.log(canvasBlocks.map((b) => b.tfFunction));
                      console.log(
                        createTfModel(canvasBlocks.map((b) => b.tfFunction())),
                      );
                    }}
                  >
                    {isTraining ? "Training..." : "Train"}
                  </button>
                </div>
                {<div>Results: {JSON.stringify(trainingRes)}</div>}
              </div>
            ) : (
              <div>History</div>
            )}
          </div>
          <div>
            <button
              className="m-2 rounded-md bg-blue-500 px-4 py-2"
              onClick={() => {
                downloadFile(
                  getConfig(
                    "pima",
                    canvasBlocks,
                    loss,
                    optimizer,
                    0.001,
                    100,
                    10,
                  ),
                );
              }}
            >
              Download Python Notebook
            </button>
          </div>
        </div>
      </div>

      <DragOverlay>
        {activeBlock && (
          <div
            style={{
              width: document
                .getElementsByClassName("overlayblock-div")[0]
                ?.getBoundingClientRect().width,
            }}
          >
            <OverlayBlock
              label={activeBlock.label}
              color={activeBlock.color}
              id={activeBlock.id}
              block={canvasBlocks.find((b) => b.id === activeBlock.id)!}
            />
          </div>
        )}
      </DragOverlay>
    </DndContext>
  );
};

export default Layers;
