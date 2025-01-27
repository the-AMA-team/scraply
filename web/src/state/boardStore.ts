import {
  DragEndEvent,
  DragOverEvent,
  DragStartEvent,
  UniqueIdentifier,
} from "@dnd-kit/core";
import { arrayMove } from "@dnd-kit/sortable";
import { createStoreWithProducer } from "@xstate/store";
import { useSelector } from "@xstate/store/react";
import { produce } from "immer";
import { BLOCKS } from "~/util/BLOCKS";
import { ActivationFunction, UILayer } from "~/types";

const boardStore = createStoreWithProducer(produce, {
  context: { canvasBlocks: [], activeBlock: null } as {
    canvasBlocks: UILayer[];
    activeBlock: UILayer | null;
  },
  on: {
    dragStart: (context, event: DragStartEvent) => {
      const { id } = event.active;
      const block =
        (BLOCKS.find((item) => item.id === id) as UILayer) ||
        (context.canvasBlocks.find(
          (item: { id: UniqueIdentifier }) => item.id === id,
        ) as UILayer);
      context.activeBlock = block;
    },

    dragOver: (context, event: DragOverEvent) => {
      const { active, over } = event;

      if (
        over &&
        active.id !== over.id &&
        context.canvasBlocks.some((block) => block.id === active.id)
      ) {
        const oldIndex = context.canvasBlocks.findIndex(
          (block) => block.id === active.id,
        );
        const newIndex = context.canvasBlocks.findIndex(
          (block) => block.id === over.id,
        );
        context.canvasBlocks = arrayMove(
          context.canvasBlocks,
          oldIndex,
          newIndex,
        );
      }
    },

    dragEnd: (context, event: DragEndEvent) => {
      const { over } = event;

      // Log the drop position
      console.log("Dropped item over target:", over);

      if (over && over.id === "canvas" && context.activeBlock) {
        const newBlock = {
          ...context.activeBlock,
          id: `${context.activeBlock.id}-${Date.now()}`,
        }; // Ensure unique ID for each new block
        context.canvasBlocks.push(newBlock);
      }

      context.activeBlock = null;
    },

    changeActivationFunction: (
      context,
      event: { id: string; activationFunction: ActivationFunction },
    ) => {
      const { id, activationFunction } = event;
      context.canvasBlocks = context.canvasBlocks.map((block) =>
        block.id === id ? { ...block, activationFunction } : block,
      );
    },

    changeNeurons: (context, event: { id: string; neurons: number }) => {
      const { id, neurons } = event;
      context.canvasBlocks = context.canvasBlocks.map((block) =>
        block.id === id ? { ...block, neurons } : block,
      );
    },
  },
});

export const useBoardStore = () => {
  return {
    canvasBlocks: useSelector(
      boardStore,
      (state) => state.context.canvasBlocks,
    ),
    activeBlock: useSelector(boardStore, (state) => state.context.activeBlock),
    changeActivationFunction: (
      id: string,
      activationFunction: ActivationFunction,
    ) => {
      boardStore.send({
        type: "changeActivationFunction",
        id,
        activationFunction,
      });
    },
    changeNeurons: (id: string, neurons: number) => {
      boardStore.send({ type: "changeNeurons", id, neurons });
    },

    drag: {
      start: (event: DragStartEvent) => {
        boardStore.send({ type: "dragStart", ...event });
      },
      over: (event: DragOverEvent) => {
        boardStore.send({ type: "dragOver", ...event });
      },
      end: (event: DragEndEvent) => {
        boardStore.send({ type: "dragEnd", ...event });
      },
    },
  };
};
