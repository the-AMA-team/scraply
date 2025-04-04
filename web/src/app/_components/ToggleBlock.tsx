import React, { useState } from "react";
import {
  IoIosArrowForward as RightArrowIcon,
  IoIosArrowDown as DownArrowIcon,
} from "react-icons/io";

interface ToggleBlockProps {
  title: React.ReactNode;
  children: React.ReactNode;
  isOpen: boolean;
  setIsOpen: React.Dispatch<React.SetStateAction<boolean>>;
  className?: string;
}

const ToggleBlock: React.FC<ToggleBlockProps> = ({
  title,
  children,
  isOpen,
  setIsOpen,
  className,
}) => {
  return (
    <div className={`p-4 ${className}`}>
      <div className={`w-full ${isOpen && "mb-4"}`}>
        <div
          className="flex w-full cursor-pointer justify-between"
          onClick={() => setIsOpen(!isOpen)}
        >
          {title}
          <div className="my-auto">
            {isOpen ? <DownArrowIcon /> : <RightArrowIcon />}
          </div>
        </div>
      </div>
      <div className={`${!isOpen && "invisible h-0"}`}>{children}</div>
    </div>
  );
};

export default ToggleBlock;
