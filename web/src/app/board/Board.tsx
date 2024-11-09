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
import { Block } from "@/types";
import OnBoardBlock from "./OnBoardBlock";
import { useArchitecture } from "@/contexts/ArchitectureContext";

const initialBlocks: Block[] = [
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
];

const getConfig = (
  input: string,
  blocks: Block[],
  loss: string,
  optimizer: string,
  learningRate: number,
  epoch: number,
  batch_size: number
) => {
  const layers = [];
  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i];

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

interface BoardProps {
  setShowConfetti: React.Dispatch<React.SetStateAction<boolean>>;
}

const Board = ({ setShowConfetti }: BoardProps) => {
  const { canvasBlocks, setCanvasBlocks } = useArchitecture()!;
  const [activeBlock, setActiveBlock] = useState<Block | null>(null);

  // running configs
  const [loss, setLoss] = useState("BCE");
  const [optimizer, setOptimizer] = useState("Adam");
  const [learningRate, setLearningRate] = useState(0.001);
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(10);

  const [isTraining, setIsTraining] = useState(false);
  const [lastLoss, setLastLoss] = useState<number | null>(null);

  console.log(canvasBlocks);

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 5,
      },
    })
  );

  const handleDragStart = (event: DragStartEvent) => {
    const { id } = event.active;
    const block =
      (initialBlocks.find((item) => item.id === id) as Block) ||
      (canvasBlocks.find(
        (item: { id: UniqueIdentifier }) => item.id === id
      ) as Block);
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
        (block) => block.id === active.id
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
      batchSize
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
        if (res.status === 200) {
          setShowConfetti(true);
        }
        return res.json();
      })
      .then((data) => {
        console.log(data);
        setLastLoss(data["final_loss"]);
      });

    setIsTraining(false);
  };

  return (
    <DndContext
      collisionDetection={closestCenter}
      onDragStart={handleDragStart}
      onDragOver={handleDragOver}
      onDragEnd={handleDragEnd}
      sensors={sensors}
    >
      <div className="flex gap-20 p-20">
        {/* Toolbox area */}
        <div className="w-36">
          <h3>Layers</h3>
          <div className="bg-zinc-800 py-1 rounded-xl">
            {initialBlocks.map((block) => (
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
          <DroppableCanvas blocks={canvasBlocks} />
        </div>

        <div className="">
          <h3>Run</h3>
          <div className="bg-zinc-800 py-1 rounded-lg p-1 text-sm px-2">
            <div className="flex my-1">
              Loss:{" "}
              <select
                className="text-white bg-zinc-700 p-1 rounded outline-none text-sm cursor-pointer mx-1"
                value={loss}
                onChange={(e) => setLoss(e.target.value)}
              >
                <option value="BCE">BCE</option>
                <option value="CrossEntropy">CrossEntropy</option>
              </select>
            </div>
            <div className="flex my-1">
              Optimizer:{" "}
              <select
                className="text-white bg-zinc-700 p-1 rounded outline-none text-sm cursor-pointer mx-1"
                value={optimizer}
                onChange={(e) => setOptimizer(e.target.value)}
              >
                <option value="Adam">Adam</option>
                <option value="AdamW">AdamW</option>
                <option value="SGD">SGD</option>
                <option value="RMSprop">RMSprop</option>
              </select>
            </div>
            <div className="flex my-1">
              Epochs:{" "}
              <input
                type="number"
                className="bg-zinc-700 rounded p-1 w-14 text-right mx-1 outline-none"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value))}
              />
            </div>
            <div className="flex my-1">
              Batch Size:{" "}
              <input
                type="number"
                className="bg-zinc-700 rounded p-1 w-14 text-right mx-1 outline-none"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value))}
              />
            </div>
            <div className="justify-center flex m-2">
              <button
                disabled={isTraining}
                className={`text-lg px-4 py-2 rounded-2xl transition-all ease-in-out bg-blue-500 ${
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
        </div>
      </div>

      <DragOverlay>
        {activeBlock && (
          <OnBoardBlock
            label={activeBlock.label}
            color={activeBlock.color}
            id={activeBlock.id}
          />
        )}
      </DragOverlay>
    </DndContext>
  );
};

export default Board;
