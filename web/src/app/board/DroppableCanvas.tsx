"use client";
import { useDroppable } from "@dnd-kit/core";
import {
  verticalListSortingStrategy,
  SortableContext,
} from "@dnd-kit/sortable";
import SortableBlock from "./SortableBlock";
import { AnyUILayer } from "../../types";
import { useBoardStore } from "~/state/boardStore";

interface DroppableCanvasProps {}

const DroppableCanvas = ({}: DroppableCanvasProps) => {
  const { canvasBlocks } = useBoardStore();
  const { setNodeRef } = useDroppable({
    id: "canvas",
  });

  return (
    <div
      ref={setNodeRef}
      className="z-10 flex h-full flex-col items-center whitespace-nowrap rounded-3xl border border-dashed border-blue-600 bg-zinc-900 p-2"
    >
      <SortableContext
        items={canvasBlocks.map((block: AnyUILayer) => block.id)}
        strategy={verticalListSortingStrategy}
      >
        {canvasBlocks.map((block) => (
          <SortableBlock
            key={block.id}
            id={block.id}
            label={block.label}
            color={block.color}
          />
        ))}
      </SortableContext>
      {/* <div className="h-10 pointer-events-none" /> Spacer at the bottom */}
    </div>
  );
};

export default DroppableCanvas;
