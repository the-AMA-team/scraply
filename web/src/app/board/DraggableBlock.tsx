"use client";
import { useDraggable } from "@dnd-kit/core";

interface DraggableBlockProps {
  id: string;
  label: string;
  color: string;
}

const DraggableBlock = ({ id, label, color }: DraggableBlockProps) => {
  const {
    active,
    attributes: _,
    listeners,
    setNodeRef,
    transform,
  } = useDraggable({
    id,
  });

  return (
    <div
      ref={setNodeRef}
      style={{
        transform: transform
          ? `translate3d(${transform.x}px, ${transform.y}px, 0)`
          : "",
        backgroundColor: color,
      }}
      {...listeners}
      // {...attributes}
      className={`m-4 cursor-grab rounded-2xl px-7 py-5 text-center ${
        active?.id == id && "opacity-0"
      }`}
    >
      {label}
    </div>
  );
};

export default DraggableBlock;
