"use client";
import { createContext, ReactNode, useContext, useState } from "react";

const DemoContext = createContext({
  isDemoing: false,
  setIsDemoing: (isDemoing: boolean) => {},
});

const DemoProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [isDemoing, setIsDemoing] = useState(false);
  return (
    <DemoContext.Provider value={{ isDemoing, setIsDemoing }}>
      {children}
    </DemoContext.Provider>
  );
};

const useDemo = () => useContext(DemoContext);

export { DemoProvider, useDemo };
