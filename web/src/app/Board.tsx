"use client";
import React, { useState } from "react";
import {
  DndContext,
  DragEndEvent,
  DragOverEvent,
  DragOverlay,
  DragStartEvent,
  closestCenter,
} from "@dnd-kit/core";
import { arrayMove } from "@dnd-kit/sortable";
import DraggableBlock from "./DraggableBlock";
import DroppableCanvas from "./DroppableCanvas";
import { Block } from "@/types";

const initialBlocks = [
  { id: "move", label: "Linear" },
  { id: "rotate", label: "Conv" },
  { id: "repeat", label: "Rnn" },
];

const ScratchLikeEditor = () => {
  const [canvasBlocks, setCanvasBlocks] = useState<Block[]>([]);
  const [activeBlock, setActiveBlock] = useState<Block | null>(null);

  const handleDragStart = (event: DragStartEvent) => {
    const { id } = event.active;
    const block =
      (initialBlocks.find((item) => item.id === id) as Block) ||
      (canvasBlocks.find((item) => item.id === id) as Block);
    setActiveBlock(block);
  };

  const handleDragOver = (event: DragOverEvent) => {
    const { active, over } = event;

    // Log to debug
    console.log("Active item:", active);
    console.log("Over target:", over);

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

  return (
    <DndContext
      collisionDetection={closestCenter}
      onDragStart={handleDragStart}
      onDragOver={handleDragOver}
      onDragEnd={handleDragEnd}
    >
      <div className="flex gap-20 p-20">
        {/* Toolbox area */}
        <div className="w-[150px]">
          <h3>Blocks</h3>
          {initialBlocks.map((block) => (
            <DraggableBlock key={block.id} id={block.id} label={block.label} />
          ))}
        </div>

        {/* Canvas area */}
        <div className="flex-grow">
          <h3>Canvas</h3>
          <DroppableCanvas blocks={canvasBlocks} />
        </div>
      </div>

      <DragOverlay>
        {activeBlock ? (
          <div className="p-8 bg-zinc-700 rounded-lg text-center">
            {activeBlock.label}
          </div>
        ) : null}
      </DragOverlay>
    </DndContext>
  );
};

export default ScratchLikeEditor;
