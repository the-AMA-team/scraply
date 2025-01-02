"use client";
import { useEffect } from "react";
import { Layer } from "../../types";
import { useDraggable } from "@dnd-kit/core";

const DraggableBlock = ({ id, label, color, activationFunction: _ }: Layer) => {
  const {
    active,
    attributes: __,
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
      className={`m-4 cursor-grab rounded-2xl p-7 text-center ${
        active?.id == id && "opacity-0"
      }`}
    >
      {label}
    </div>
  );
};

export default DraggableBlock;
