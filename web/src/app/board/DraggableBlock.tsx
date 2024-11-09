"use client";
import { Block } from "@/types";
import { useDraggable } from "@dnd-kit/core";

const DraggableBlock = ({ id, label, color }: Block) => {
  const { active, attributes, listeners, setNodeRef, transform } = useDraggable(
    {
      id,
    }
  );

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
      {...attributes}
      className={`p-8 m-4 rounded-2xl text-center cursor-grab ${
        active?.id == id && "opacity-0"
      }`}
    >
      {label}
    </div>
  );
};

export default DraggableBlock;
