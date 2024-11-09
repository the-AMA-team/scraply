import React from "react";

interface OnBoardBlockProps {
  label: string;
  color: string;
}

const OnBoardBlock = ({ label, color }: OnBoardBlockProps) => {
  return (
    <div
      className={`px-8 py-20 bg-zinc-800 rounded-lg text-center bg-${color}`}
    >
      <div>{label}</div>
    </div>
  );
};

export default OnBoardBlock;
