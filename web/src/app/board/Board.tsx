"use client";
import React, { useEffect, useState } from "react";
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
import axios from "axios";
import { useUser } from "@/contexts/UserContext";
import { Attempt, Level } from "@prisma/client";
import { useRouter } from "next/navigation";
import { ResponsiveLine } from "@nivo/line";

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

const getUserProgress = async (
  level: number,
  uid: string
): Promise<Attempt | undefined> => {
  try {
    const response = await axios.get(
      `http://localhost:3000/api/attempt/${uid}`
    );
    console.log(response);
    console.log(response.data);
    if (response.data) {
      return response.data.attempts.find(
        (attempt: any) => attempt.level === `L${level}`
      );
    }
  } catch (error) {
    console.error(error);
  }
};

const updateUserProgress = async (
  userId: string,
  attemptId: string,
  updatedData: Omit<Omit<Attempt, "id">, "userId">
) => {
  console.log("updateUserProgress");
  try {
    const response = await axios.post("http://localhost:3000/api/attempt", {
      userId,
      attemptId,
      updatedData,
    });
    console.log(response);
  } catch (error) {
    console.error(error);
  }
};

interface BoardProps {
  level: number;
  setShowConfetti: React.Dispatch<React.SetStateAction<boolean>>;
}

const Board = ({ level, setShowConfetti }: BoardProps) => {
  const { user, loading } = useUser()!;
  const router = useRouter();

  if (!user && !loading) {
    router.push("/");
  }

  const [currentAttmpt, setCurrentAttempt] = useState<Attempt | null>(null);

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
  const [trainingRes, setTrainingRes] = useState<any | null>(null);
  const [progress, setProgress] = useState(0);

  const [isModalOpen, setIsModalOpen] = useState(true);

  const [graphData, setGraphData] = useState<any | null>(null);

  console.log(canvasBlocks);

  useEffect(() => {
    // get user progress
    if (!user) return;
    getUserProgress(level, user.uid).then((attempt) => {
      // convert the json in attempt.archetecture to blocks
      if (attempt?.archetecture) {
        try {
          const jsonLayers = JSON.parse(attempt.archetecture as string).layers;

          const blocks = [];

          for (let i = 0; i < jsonLayers.length; i += 2) {
            const b_: Block = {
              id: `${jsonLayers[i].kind}-${Math.random()}`,
              label: jsonLayers[i].kind,
              neurons: jsonLayers[i].args[0],
              color:
                initialBlocks.find((item) => item.label === jsonLayers[i].kind)
                  ?.color || "#20FF8F",
              activationFunction: jsonLayers[i + 1].kind,
            };
            blocks.push(b_);
          }
          console.log(blocks);
          setCanvasBlocks(blocks);
        } catch (error) {
          console.error(error);
        }
      }
      setCurrentAttempt(attempt || null);

      setLastLoss(attempt?.lastLoss || null);
    });
  }, [user]);

  useEffect(() => {
    console.log(currentAttmpt);
  }, [currentAttmpt]);

  useEffect(() => {
    // update user progress
    if (!user || !currentAttmpt) return;
    if (lastLoss) {
      updateUserProgress(user.uid, currentAttmpt!.id, {
        level: `L${level}` as Level,
        rating: 1,
        archetecture: JSON.stringify(
          getConfig(
            "pima",
            canvasBlocks,
            loss,
            optimizer,
            learningRate,
            epochs,
            batchSize
          )
        ),
        lastLoss,
      });
    }
  }, [lastLoss]);

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
        // setLastLoss(data["final_loss"]);
        setTrainingRes(data);
        setProgress(Math.round(data["avg_test_acc"] * 100) * 0.01);

        const testLosses = data["test_losses"].map(
          (item: number, idx: number) => ({
            x: idx,
            y: item,
          })
        );

        const trainLosses = data["train_losses"].map(
          (item: number, idx: number) => ({
            x: idx,
            y: item,
          })
        );

        setGraphData([
          { id: "test_loss", data: testLosses },
          { id: "train_loss", data: trainLosses },
        ]);
      });

    setIsTraining(false);
    setIsModalOpen(true);
  };

  useEffect(() => {
    // document.getElementById("upload_modal")?.showModal();
  }, []);

  // Calculate the stroke-dasharray for the progress circle
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
        className={`absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 h-1/2 w-1/2 bg-zinc-800 z-20 rounded-2xl shadow-xl ${
          !isModalOpen && "hidden"
        }`}
      >
        <div className="flex justify-center my-4">
          <div className="w-80 rounded-full">
            <div className="flex flex-col items-center relative">
              {/* Progress Circle Container */}
              <svg className="w-52 h-52 relative" viewBox="0 0 36 36">
                {/* Background Circle */}
                <circle
                  className="text-zinc-700"
                  strokeWidth="3"
                  stroke="currentColor"
                  fill="transparent"
                  r={circleRadius}
                  cx="18"
                  cy="18"
                />
                {/* Progress Circle */}
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

              {/* Centered Image */}
              <img
                src="/stars.png"
                alt="centered-icon"
                className="absolute w-60 top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 rounded-full"
              />

              {/* Display Progress Percentage */}
              <div className="mt-2 text-center text-lg font-semibold text-green-500">
                {progress}% Accuracy Achieved!
              </div>
              {/* {JSON.stringify(trainingRes)} */}
            </div>
          </div>
        </div>
        <div className="flex justify-center mt-4 text-zinc-400">
          Suggested change for higher accuracy: Increase the number of neurons
          in layer 2.
        </div>
        <div className="flex justify-center mt-4">
          <button
            onClick={() => {
              setIsModalOpen(false);
            }}
            className="bg-blue-500 text-white px-4 py-2 rounded"
          >
            YAY!
          </button>
        </div>
      </div>
      <div className={`flex gap-20 p-20 pt-10 ${isModalOpen && "opacity-50"}`}>
        {/* Toolbox area */}
        <div className="w-36">
          <h3>Scraps</h3>
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
          <div>
            {trainingRes?.test_losses && (
              <div className="h-72 w-72 mt-2 text-center bg-zinc-50 rounded-md">
                <ResponsiveLine
                  data={graphData}
                  margin={{ top: 50, right: 110, bottom: 50, left: 60 }}
                  enableGridX={false}
                  enableGridY={false}
                  xScale={{ type: "point" }}
                  yScale={{
                    type: "linear",
                    min: "auto",
                    max: "auto",
                    stacked: true,
                    reverse: false,
                  }}
                  axisTop={null}
                  axisRight={null}
                  axisBottom={null}
                  axisLeft={{
                    tickSize: 5,
                    tickPadding: 5,
                    tickRotation: 0,
                    legend: "Loss",
                    legendOffset: -40,
                    legendPosition: "middle",
                  }}
                  // colors={{ scheme: "" }}
                  pointSize={2}
                  pointColor={{ theme: "background" }}
                  pointBorderWidth={2}
                  pointBorderColor={{ from: "serieColor" }}
                  pointLabelYOffset={-12}
                  useMesh={true}
                  legends={[
                    {
                      anchor: "bottom-right",
                      direction: "column",
                      justify: false,
                      translateX: 100,
                      translateY: 0,
                      itemsSpacing: 0,
                      itemDirection: "left-to-right",
                      itemWidth: 80,
                      itemHeight: 20,
                      itemOpacity: 0.75,
                      symbolSize: 12,
                      symbolShape: "circle",
                      symbolBorderColor: "rgba(0, 0, 0, .5)",
                    },
                  ]}
                />
              </div>
            )}
            <button className="bg-blue-500 py-2 px-4 m-2 rounded-md">
              Download Python Notebook
            </button>
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
