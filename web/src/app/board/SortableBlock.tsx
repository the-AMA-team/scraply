"use client";
import { UILayer } from "../../types";
import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import OverlayBlock from "./OverlayBlock";

interface SortableBlockProps {
  id: string;
  label: string;
  color: string;
  layers: UILayer[];
  setLayers: React.Dispatch<React.SetStateAction<UILayer[]>>;
}

const SortableBlock = ({
  id,
  label,
  color,
  layers,
  setLayers,
}: SortableBlockProps) => {
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
        block={layers.find((block) => block.id === id)!}
        setCanvasBlocks={setLayers}
      />
    </div>
  );
};

export default SortableBlock;
