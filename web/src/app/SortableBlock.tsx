import { Block } from "@/types";
import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import OnBoardBlock from "./OnBoardBlock";

const SortableBlock = ({ id, label, color }: Block) => {
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
      className="mx-2"
    >
      <OnBoardBlock label={label} color={color} />
    </div>
  );
};

export default SortableBlock;
