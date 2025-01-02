"use client";
import React, { useState } from "react";
import {
  DndContext,
  DragEndEvent,
  DragOverEvent,
  DragOverlay,
  DragStartEvent,
  PointerSensor,
  UniqueIdentifier,
  closestCenter,
  useSensor,
  useSensors,
} from "@dnd-kit/core";
import { arrayMove } from "@dnd-kit/sortable";
import DraggableBlock from "./DraggableBlock";
import DroppableCanvas from "./DroppableCanvas";
import { Layer } from "../../types";
import OverlayBlock from "./OverlayBlock";

const BLOCKS: Layer[] = [
  {
    id: "linear",
    label: "Linear",
    color: "#20FF8F",
    activationFunction: "ReLU",
    neurons: 8,
  },
  {
    id: "conv",
    label: "Conv",
    color: "#FFD620",
    activationFunction: "ReLU",
    neurons: 8,
  },
  {
    id: "rnn",
    label: "RNN",
    color: "#FF8C20",
    activationFunction: "ReLU",
    neurons: 8,
  },
  {
    id: "gru",
    label: "GRU",
    color: "#FF4920",
    activationFunction: "ReLU",
    neurons: 8,
  },
  {
    id: "flatten",
    label: "Flatten",
    color: "#FF208F",
    activationFunction: "ReLU",
    neurons: 8,
  },
];

const getConfig = (
  input: string,
  blocks: Layer[],
  loss: string,
  optimizer: string,
  learningRate: number,
  epoch: number,
  batch_size: number,
) => {
  const layers = [];
  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i]!;

    const currentNeuron = block.neurons;
    const nextNeuron = blocks[i + 1]?.neurons || 1; // Default to 1 if no next block, could change based on the dataset
    layers.push({
      kind: block.label,
      args: [currentNeuron, nextNeuron],
    });
    layers.push({
      kind: block.activationFunction,
    });
  }

  const config = {
    input,
    layers,
    loss,
    optimizer: { kind: optimizer, lr: learningRate },
    epoch,
    batch_size,
  };

  return config;
};

const downloadFile = async (config: any) => {
  await fetch("http://127.0.0.1:5000/generate", {
    method: "POST",
    body: JSON.stringify(config),
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.blob();
    })
    .then((blob) => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.style.display = "none";
      a.href = url;
      a.download = "generated_notebook.ipynb";

      document.body.appendChild(a);
      a.click();

      // clean up
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    })
    .catch((error) => {
      console.error("Error downloading file:", error);
    });
};

interface BoardProps {}

const Board = (props: BoardProps) => {
  const [canvasBlocks, setCanvasBlocks] = useState<Layer[]>([]);

  const [activeBlock, setActiveBlock] = useState<Layer | null>(null);

  // running configs
  const [loss, setLoss] = useState("BCE");
  const [optimizer, setOptimizer] = useState("Adam");
  const [learningRate, setLearningRate] = useState(0.001);
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(10);

  const [isTraining, setIsTraining] = useState(false);
  const [lastLoss, setLastLoss] = useState<number | null>(null);
  const [trainingRes, setTrainingRes] = useState<any | null>(null);
  const [progress, setProgress] = useState(0);

  const [isModalOpen, setIsModalOpen] = useState(false);

  const [advice, setAdvice] = useState("");

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 5,
      },
    }),
  );

  const handleDragStart = (event: DragStartEvent) => {
    const { id } = event.active;
    const block =
      (BLOCKS.find((item) => item.id === id) as Layer) ||
      (canvasBlocks.find(
        (item: { id: UniqueIdentifier }) => item.id === id,
      ) as Layer);
    setActiveBlock(block);
  };

  const handleDragOver = (event: DragOverEvent) => {
    const { active, over } = event;

    if (
      over &&
      active.id !== over.id &&
      canvasBlocks.some((block) => block.id === active.id)
    ) {
      const oldIndex = canvasBlocks.findIndex(
        (block) => block.id === active.id,
      );
      const newIndex = canvasBlocks.findIndex((block) => block.id === over.id);
      setCanvasBlocks((blocks) => arrayMove(blocks, oldIndex, newIndex));
    }
  };

  const handleDragEnd = (event: DragEndEvent) => {
    const { over } = event;

    // Log the drop position
    console.log("Dropped item over target:", over);

    if (over && over.id === "canvas" && activeBlock) {
      const newBlock = {
        ...activeBlock,
        id: `${activeBlock.id}-${Date.now()}`,
      }; // Ensure unique ID for each new block
      setCanvasBlocks((prevBlocks) => [...prevBlocks, newBlock]);
    }

    setActiveBlock(null);
  };

  const startTraining = async () => {
    setIsTraining(true);
    const config = getConfig(
      "pima",
      canvasBlocks,
      loss,
      optimizer,
      learningRate,
      epochs,
      batchSize,
    );
    console.log(config);

    await fetch("http://127.0.0.1:5000/train", {
      method: "POST",
      body: JSON.stringify(config),
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((res) => {
        return res.json();
      })
      .then((data) => {
        console.log(data);
        setLastLoss(data["final_loss"]);
        setTrainingRes(data.RESULTS);
        setProgress(Math.round(data.RESULTS["avg_test_acc"] * 100) * 0.01);
        setAdvice(data.sad_advice_string);
      });

    setIsTraining(false);
    setIsModalOpen(true);
  };

  const circleRadius = 16;
  const circumference = 2 * Math.PI * circleRadius;
  const dashArray = `${(progress / 100) * circumference} ${circumference}`;

  return (
    <DndContext
      collisionDetection={closestCenter}
      onDragStart={handleDragStart}
      onDragOver={handleDragOver}
      onDragEnd={handleDragEnd}
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
                {progress}% Accuracy Achieved!
              </div>
            </div>
          </div>
        </div>
        <div className="mt-4 flex justify-center px-8 text-center text-blue-500">
          Advice: {advice}
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
      <div className={`flex gap-20 p-20 pt-10 ${isModalOpen && "opacity-50"}`}>
        {/* Toolbox area */}
        <div className="w-36">
          <h3>Scraps</h3>
          <div className="rounded-xl bg-zinc-800 py-1">
            {BLOCKS.map((block) => (
              <DraggableBlock
                key={block.id}
                id={block.id}
                label={block.label}
                color={block.color}
                activationFunction={block.activationFunction}
                neurons={block.neurons}
              />
            ))}
          </div>
        </div>

        {/* Canvas area */}
        <div className="flex-grow">
          <h3>Canvas</h3>
          <DroppableCanvas
            layers={canvasBlocks}
            setCanvasBlocks={setCanvasBlocks}
          />
        </div>

        <div className="">
          <h3>Run</h3>
          <div className="rounded-lg bg-zinc-800 p-1 px-2 py-1 text-sm">
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
                type="number"
                className="mx-1 w-14 rounded bg-zinc-700 p-1 text-right outline-none"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value))}
              />
            </div>
            <div className="my-1 flex">
              Batch Size:{" "}
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
                onClick={startTraining}
              >
                {isTraining ? "Training..." : "Train"}
              </button>
            </div>
            {<div>Last Loss: {lastLoss}</div>}
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
          <OverlayBlock
            label={activeBlock.label}
            color={activeBlock.color}
            id={activeBlock.id}
            canvasBlocks={canvasBlocks}
            setCanvasBlocks={setCanvasBlocks}
          />
        )}
      </DragOverlay>
    </DndContext>
  );
};

export default Board;
