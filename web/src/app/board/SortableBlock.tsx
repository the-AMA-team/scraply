"use client";
import { UILayer } from "../../types";
import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import OverlayBlock from "./OverlayBlock";
import { useBoardStore } from "~/state/boardStore";

interface SortableBlockProps {
  id: string;
  label: string;
  color: string;
}

const SortableBlock = ({ id, label, color }: SortableBlockProps) => {
  const { canvasBlocks } = useBoardStore();
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      {...listeners}
      {...attributes}
      className=""
    >
      <OverlayBlock
        label={label}
        color={color}
        id={id}
        block={canvasBlocks.find((block) => block.id === id)!}
      />
    </div>
  );
};

export default SortableBlock;
