"use client";
import { useDroppable } from "@dnd-kit/core";
import {
  horizontalListSortingStrategy,
  SortableContext,
} from "@dnd-kit/sortable";
import SortableBlock from "./SortableBlock";
import { UILayer } from "../../types";
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
      className="z-10 flex h-5/6 items-center whitespace-nowrap rounded-3xl border-2 border-dashed border-zinc-600 bg-zinc-900 p-8 px-28"
    >
      <SortableContext
        items={canvasBlocks.map((block: UILayer) => block.id)}
        strategy={horizontalListSortingStrategy}
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
