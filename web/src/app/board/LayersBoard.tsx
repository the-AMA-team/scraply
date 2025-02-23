import {
  useSensors,
  useSensor,
  PointerSensor,
  DndContext,
  closestCenter,
  DragOverlay,
} from "@dnd-kit/core";
import React, { useEffect, useState } from "react";
import { useBoardStore } from "~/state/boardStore";
import {
  startTraining,
  downloadFile,
  getConfig,
  getArchitectureSuggestion,
} from "~/util/board.util";
import { LAYER_BLOCKS } from "../../util/LAYER_BLOCKS";
import DraggableBlock from "./DraggableBlock";
import DroppableCanvas from "./DroppableCanvas";
import OverlayBlock from "./OverlayBlock";

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
  isLoadingSuggestionsState: [
    boolean,
    React.Dispatch<React.SetStateAction<boolean>>,
  ];
}

const LayersBoard = ({
  lossState,
  optimizerState,
  learningRateState,
  epochState,
  batchSizeState,
  isTrainingState,
  trainingResState,
  progressState,
  isLoadingSuggestionsState,
}: LayersProps) => {
  const { canvasBlocks, setCanvasBlocks, activeBlock, drag } = useBoardStore();
  const [loss, setLoss] = lossState;
  const [optimizer, setOptimizer] = optimizerState;
  const [learningRate, setLearningRate] = learningRateState;
  const [epochs, setEpochs] = epochState;
  const [batchSize, setBatchSize] = batchSizeState;
  const [isTraining, setIsTraining] = isTrainingState;
  const [trainingRes, setTrainingRes] = trainingResState;
  const [progress, setProgress] = progressState;

  const [isLoadingSuggestions, setIsLoadingSuggestions] =
    isLoadingSuggestionsState;

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
      <div className={`mx-20 mt-10 flex ${isModalOpen && "opacity-50"}`}>
        {/* Canvas area */}
        <div className="mr-10 flex-grow">
          <div className="bg-zinc-900 p-2 text-2xl text-zinc-500">Canvas</div>
          <div className="relative">
            <DroppableCanvas />
            <button
              title="Get suggested architecture using AI"
              className={`absolute bottom-4 right-4 rounded-lg bg-blue-500 px-3 py-2 shadow-lg ring-indigo-500 ring-offset-2 ring-offset-zinc-900 transition-all duration-100 ${isLoadingSuggestions ? "w-16 bg-indigo-500 p-0" : "hover:bg-indigo-600 hover:ring-2 active:bg-indigo-500"}`}
              disabled={isLoadingSuggestions}
              onClick={() => {
                setIsLoadingSuggestions(true);
                getArchitectureSuggestion("pima")
                  .then((data) => {
                    const layers = data.map((l) => {
                      return {
                        id: `${l.label}-${Math.random()}`,
                        label: l.label,
                        color: l.color,
                        neurons: l.neurons,
                        otherParam: l.otherParam,
                        activationFunction: l.activationFunction,
                      };
                    });
                    setCanvasBlocks(layers);
                  })
                  .finally(() => {
                    setIsLoadingSuggestions(false);
                  });
              }}
            >
              {isLoadingSuggestions ? (
                <img src="dino-running.gif" className="w-14" />
              ) : (
                "✨"
              )}
            </button>
          </div>
        </div>

        {/* Toolbox area */}
        <div className="mr-4">
          <div className="bg-zinc-900 p-2 text-center text-2xl text-zinc-500">
            Scraps
          </div>
          <div className="rounded-xl bg-zinc-800 py-1">
            {LAYER_BLOCKS.map((block) => (
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
          <div className="bg-zinc-900 p-2 text-2xl text-zinc-500">Train</div>
          <div className="rounded-lg bg-zinc-800 p-1 px-2 py-1 text-sm">
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
                Learning Rate:{" "}
                <input
                  type="range"
                  name="Learning Rate"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                  min={0.001}
                  max={0.1}
                  step={0.001}
                />
                <input
                  type="number"
                  className="mx-1 w-14 rounded bg-zinc-700 py-1 text-right outline-none"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                />
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
                  className="mx-1 w-14 rounded bg-zinc-700 py-1 text-right outline-none"
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
                  className="mx-1 w-14 rounded bg-zinc-700 py-1 text-right outline-none"
                  value={batchSize}
                  onChange={(e) => setBatchSize(parseInt(e.target.value))}
                />
              </div>
              <div
                className={`m-2 ${isTraining && "mt-8"} flex justify-center transition-all duration-300`}
              >
                <button
                  disabled={isTraining}
                  className={`rounded-2xl bg-zinc-700 px-6 py-2 text-lg transition-all ease-in-out ${
                    !isTraining &&
                    "hover:bg-indigo-600 hover:px-8 hover:ring-2 active:bg-indigo-500 active:px-9"
                  } ring-indigo-500 duration-300 ${
                    isTraining && "animate-bounce px-9 ring-2 ring-zinc-600"
                  }`}
                  onClick={() => {
                    setIsTraining(true);

                    startTraining(
                      getConfig(
                        "pima",
                        canvasBlocks,
                        loss,
                        optimizer,
                        learningRate,
                        epochs,
                        batchSize,
                      ),
                    ).then((data: any) => {
                      setTrainingRes(data.RESULTS);
                      setProgress(
                        Math.round(data.RESULTS["avg_test_acc"] * 100) * 0.01,
                      );
                      setIsTraining(false);
                      setIsModalOpen(true);
                    });
                  }}
                >
                  {isTraining ? (
                    <div className="flex items-center">
                      <div>Training</div>{" "}
                      <img src="dino-running.gif" className="w-14" />
                    </div>
                  ) : (
                    "Train"
                  )}
                </button>
              </div>
            </div>
          </div>
          <div>
            <div>
              <button
                className="my-2 w-full rounded-md bg-blue-500 px-4 py-2"
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

export default LayersBoard;
