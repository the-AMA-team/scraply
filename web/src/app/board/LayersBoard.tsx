import {
  useSensors,
  useSensor,
  PointerSensor,
  DndContext,
  closestCenter,
  DragOverlay,
} from "@dnd-kit/core";
import React, { useEffect, useRef, useState } from "react";
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
import DemoCard from "../_components/DemoCard";
import { useDemo } from "~/state/DemoContext";
import ToggleBlock from "../_components/ToggleBlock";
import { CgSpinnerTwoAlt as SpinnerIcon } from "react-icons/cg";

const DEMO_CARDS = [
  {
    title: "Canvas Area",
    description: `Drag and drop layers ("scraps") here to build your model. Drag and drop layers ("scraps") here to build your model. Drag and drop layers ("scraps") here to build your model.`,
    x: 0,
    y: 0,
    section: "canvas",
  },
  {
    title: "Scraps Area",
    description: `Drag and drop layers ("scraps") here to build your model. Drag and drop layers ("scraps") here to build your model. Drag and drop layers ("scraps") here to build your model.`,
    x: 0,
    y: 0,
    section: "scraps",
  },
  {
    title: "Training Area",
    description: `Drag and drop layers ("scraps") here to build your model. Drag and drop layers ("scraps") here to build your model. Drag and drop layers ("scraps") here to build your model.`,
    x: 0,
    y: 0,
    section: "training",
  },
];

interface LayersProps {
  selectedDataset: string;
  trainingBlockToggleState: [
    boolean,
    React.Dispatch<React.SetStateAction<boolean>>,
  ];
  lossState: [string, React.Dispatch<React.SetStateAction<string>>];
  optimizerState: [string, React.Dispatch<React.SetStateAction<string>>];
  learningRateState: [number, React.Dispatch<React.SetStateAction<number>>];
  epochState: [number, React.Dispatch<React.SetStateAction<number>>];
  batchSizeState: [number, React.Dispatch<React.SetStateAction<number>>];
  isTrainingState: [boolean, React.Dispatch<React.SetStateAction<boolean>>];
  trainingResHistoryState: [any[], React.Dispatch<React.SetStateAction<any[]>>];
  progressState: [number, React.Dispatch<React.SetStateAction<number>>];
  isLoadingSuggestionsState: [
    boolean,
    React.Dispatch<React.SetStateAction<boolean>>,
  ];
  resultsBlockToggleState: [
    boolean,
    React.Dispatch<React.SetStateAction<boolean>>,
  ];
  showNotification: (title: string, body: string) => void;
}

const LayersBoard = ({
  selectedDataset,
  trainingBlockToggleState,
  lossState,
  optimizerState,
  learningRateState,
  epochState,
  batchSizeState,
  isTrainingState,
  trainingResHistoryState,
  progressState,
  isLoadingSuggestionsState,
  resultsBlockToggleState,
  showNotification,
}: LayersProps) => {
  const boardRef = useRef<HTMLDivElement>(null);
  const canvasSectionRef = useRef<HTMLDivElement>(null);
  const scrapsSectionRef = useRef<HTMLDivElement>(null);
  const trainingSectionRef = useRef<HTMLDivElement>(null);

  const SectionRefs = {
    canvas: canvasSectionRef,
    scraps: scrapsSectionRef,
    training: trainingSectionRef,
  };

  const { isDemoing, setIsDemoing } = useDemo();
  const [demoIdx, setDemoIdx] = useState(0);

  useEffect(() => {
    if (isDemoing) {
      Array.from(boardRef.current?.children).forEach((child) => {
        child.classList.add("opacity-20");
      });
      SectionRefs[DEMO_CARDS[demoIdx]?.section!].current?.classList.remove(
        "opacity-20",
      );
    }
  }, [isDemoing, demoIdx]);

  const { canvasBlocks, setCanvasBlocks, activeBlock, drag } = useBoardStore();

  const [isTrainingBlockOpen, setIsTrainingBlockOpen] =
    trainingBlockToggleState;

  const [loss, setLoss] = lossState;
  const [optimizer, setOptimizer] = optimizerState;
  const [learningRate, setLearningRate] = learningRateState;
  const [epochs, setEpochs] = epochState;
  const [batchSize, setBatchSize] = batchSizeState;
  const [isTraining, setIsTraining] = isTrainingState;
  const [trainingResHistory, setTrainingResHistory] = trainingResHistoryState;
  const [progress, setProgress] = progressState;

  const [isLoadingSuggestions, setIsLoadingSuggestions] =
    isLoadingSuggestionsState;

  const [isResultsBlockOpen, setIsResultsBlockOpen] = resultsBlockToggleState;

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
      <div className={`mx-20 mt-10 flex`} ref={boardRef}>
        {/* Toolbox area */}
        <div className="mr-4" ref={scrapsSectionRef}>
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
        {/* Canvas area */}
        <div className="mr-10 flex-grow" ref={canvasSectionRef}>
          <div className="relative">
            <DroppableCanvas />
            <button
              title="Get suggested architecture using AI"
              className={`absolute bottom-4 right-4 rounded-lg bg-blue-500 px-3 py-2 shadow-lg ring-indigo-500 ring-offset-2 ring-offset-zinc-900 transition-all duration-100 ${isLoadingSuggestions ? "w-16 bg-indigo-500 p-0" : "hover:bg-indigo-600 hover:ring-2 active:bg-indigo-500"}`}
              disabled={isLoadingSuggestions}
              onClick={() => {
                setIsLoadingSuggestions(true);
                getArchitectureSuggestion(selectedDataset)
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
              {isLoadingSuggestions
                ? // <img src="dino-running.gif" className="w-14" />
                  "..."
                : "✨"}
            </button>
          </div>
        </div>

        {/* Training config */}
        <div className="" ref={trainingSectionRef}>
          <ToggleBlock
            isOpen={isTrainingBlockOpen}
            setIsOpen={setIsTrainingBlockOpen}
            title={
              <div className="flex w-full justify-between">
                <div className="text-xl">Training Config</div>
                {!isTrainingBlockOpen && (
                  <button
                    className={`mx-2 my-auto rounded-lg bg-zinc-700 px-4 py-1 transition-colors ease-in-out ${
                      !isTraining && "hover:bg-indigo-600 active:bg-indigo-500"
                    } ring-indigo-500 duration-300`}
                    onClick={() => {
                      setIsTraining(true);

                      startTraining(
                        getConfig(
                          selectedDataset,
                          canvasBlocks,
                          loss,
                          optimizer,
                          learningRate,
                          epochs,
                          batchSize,
                        ),
                      )
                        .then((data: any) => {
                          setTrainingResHistory([
                            data.RESULTS,
                            ...trainingResHistory,
                          ]);
                          setProgress(
                            Math.round(data.RESULTS["avg_test_acc"] * 100) *
                              0.01,
                          );
                        })
                        .finally(() => {
                          setIsTraining(false);
                          showNotification(
                            "Training Complete!",
                            "Your model has been trained successfully.",
                          );
                        });
                    }}
                  >
                    {isTraining ? (
                      <SpinnerIcon className="h-5 animate-spin" />
                    ) : (
                      "Train"
                    )}
                  </button>
                )}
              </div>
            }
            className={`rounded-xl ring ${isTraining ? "ring-2 ring-orange-500" : "ring-zinc-700"}`}
          >
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
                    onChange={(e) =>
                      setLearningRate(parseFloat(e.target.value))
                    }
                    min={0.001}
                    max={0.1}
                    step={0.001}
                  />
                  <input
                    type="number"
                    className="mx-1 w-14 rounded bg-zinc-700 py-1 text-right outline-none"
                    value={learningRate}
                    onChange={(e) =>
                      setLearningRate(parseFloat(e.target.value))
                    }
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
                  className={`m-2 ${isTraining && "mt-8"} flex justify-center transition-transform duration-300`}
                >
                  <button
                    disabled={isTraining}
                    className={`rounded-2xl bg-zinc-700 px-6 py-2 text-lg transition-colors ease-in-out ${
                      !isTraining && "hover:bg-indigo-600 active:bg-indigo-500"
                    } ring-indigo-500 duration-300 ${
                      isTraining && "px-9 ring-2 ring-zinc-600"
                    }`}
                    onClick={() => {
                      setIsTraining(true);

                      startTraining(
                        getConfig(
                          selectedDataset,
                          canvasBlocks,
                          loss,
                          optimizer,
                          learningRate,
                          epochs,
                          batchSize,
                        ),
                      )
                        .then((data: any) => {
                          setTrainingResHistory([
                            data.RESULTS,
                            ...trainingResHistory,
                          ]);
                          setProgress(
                            Math.round(data.RESULTS["avg_test_acc"] * 100) *
                              0.01,
                          );
                        })
                        .finally(() => {
                          setIsTraining(false);
                          showNotification(
                            "Training Complete!",
                            "Your model has been trained successfully.",
                          );
                        });
                    }}
                  >
                    {isTraining ? (
                      <div className="flex items-center">
                        <div className="flex">
                          <SpinnerIcon className="my-auto mr-2 h-5 animate-spin" />
                          <div>Training...</div>
                        </div>
                        {/* <img src="dino-running.gif" className="w-14" /> */}
                      </div>
                    ) : (
                      "Train"
                    )}
                  </button>
                </div>
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
          </ToggleBlock>
          <ToggleBlock
            isOpen={isResultsBlockOpen}
            setIsOpen={setIsResultsBlockOpen}
            className="mt-4 h-1/2 rounded-xl ring ring-zinc-700"
            title={
              <div className="flex w-full justify-between">
                <div className="text-xl">Results</div>
              </div>
            }
          >
            <div className="text-sm">
              {trainingResHistory.length !== 0 ? (
                trainingResHistory.map((trainingRes, idx) => {
                  return (
                    <div
                      key={idx}
                      className={`rounded-lg bg-zinc-800 p-1 px-2 py-1 ${isResultsBlockOpen && "my-2"}`}
                    >
                      <div>#{trainingResHistory.length - idx}</div>
                      {Object.keys(trainingRes).map((key) => {
                        return (
                          <div key={key} className="flex justify-between">
                            <div>{key}</div>
                            <div className="flex">
                              {idx !== trainingResHistory.length - 1 &&
                                (() => {
                                  const diff = Math.round(
                                    trainingRes[key] -
                                      trainingResHistory[idx + 1][key],
                                  );
                                  return diff < 0 ? (
                                    <div className="text-zinc-500">
                                      (
                                      <span className="text-red-600">
                                        {diff}
                                      </span>
                                      )
                                    </div>
                                  ) : (
                                    <div className="text-zinc-500">
                                      (
                                      <span className="text-green-600">
                                        +{diff}
                                      </span>
                                      )
                                    </div>
                                  );
                                })()}
                              <div className="ml-1">
                                {Math.round(trainingRes[key])}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  );
                })
              ) : (
                <div className="text-center">No training results yet.</div>
              )}
            </div>
          </ToggleBlock>
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

      {isDemoing && (
        <div className="absolute left-0 top-0 z-20 flex h-full w-full items-center justify-center bg-zinc-900 bg-opacity-0">
          <DemoCard
            currIdx={demoIdx}
            description={DEMO_CARDS[demoIdx]?.description!}
            maxIdx={DEMO_CARDS.length - 1}
            next={demoIdx !== DEMO_CARDS.length - 1}
            prev={demoIdx !== 0}
            onNext={() => setDemoIdx(demoIdx + 1)}
            onPrev={() => {
              setDemoIdx(demoIdx - 1);
            }}
            title={DEMO_CARDS[demoIdx]?.title!}
            x={DEMO_CARDS[demoIdx]?.x!}
            y={DEMO_CARDS[demoIdx]?.y!}
            component={SectionRefs[DEMO_CARDS[demoIdx]?.section!]}
            closeDemo={() => {
              setIsDemoing(false);
              Array.from(boardRef.current?.children).forEach((child) => {
                child.classList.remove("opacity-20");
              });
              setDemoIdx(0);
            }}
          />
        </div>
      )}
    </DndContext>
  );
};

export default LayersBoard;
