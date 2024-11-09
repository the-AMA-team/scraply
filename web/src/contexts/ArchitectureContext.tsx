"use client";
import { Block } from "@/types";
import { useContext, createContext, useState } from "react";

interface ArchitectureContextType {
  canvasBlocks: Block[];
  setCanvasBlocks: React.Dispatch<React.SetStateAction<Block[]>>;
}

const ArchitectureContext = createContext<ArchitectureContextType | undefined>(
  undefined
);

export const useArchitecture = () => useContext(ArchitectureContext);

export const ArchitectureProvider = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  const [canvasBlocks, setCanvasBlocks] = useState<Block[]>([]);

  return (
    <ArchitectureContext.Provider
      value={{
        canvasBlocks: canvasBlocks,
        setCanvasBlocks: setCanvasBlocks,
      }}
    >
      {children}
    </ArchitectureContext.Provider>
  );
};
