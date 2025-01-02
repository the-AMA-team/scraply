"use client";
import { useDroppable } from "@dnd-kit/core";
import {
  horizontalListSortingStrategy,
  SortableContext,
} from "@dnd-kit/sortable";
import SortableBlock from "./SortableBlock";
import { Layer } from "../../types";

interface DroppableCanvasProps {
  layers: Layer[];
  setCanvasBlocks: React.Dispatch<React.SetStateAction<Layer[]>>;
}

const DroppableCanvas = ({
  layers: blocks,
  setCanvasBlocks,
}: DroppableCanvasProps) => {
  const { setNodeRef } = useDroppable({
    id: "canvas",
  });

  return (
    <div
      ref={setNodeRef}
      className="z-10 flex h-5/6 items-center whitespace-nowrap rounded-3xl border-2 border-dashed border-zinc-600 bg-zinc-900 p-8 px-28"
    >
      <SortableContext
        items={blocks.map((block: Layer) => block.id)}
        strategy={horizontalListSortingStrategy}
      >
        {blocks.map((block) => (
          <SortableBlock
            key={block.id}
            id={block.id}
            label={block.label}
            color={block.color}
            activationFunction={block.activationFunction}
            neurons={block.neurons}
            layers={blocks}
            setLayers={setCanvasBlocks}
          />
        ))}
      </SortableContext>
      {/* <div className="h-10 pointer-events-none" /> Spacer at the bottom */}
    </div>
  );
};

export default DroppableCanvas;
