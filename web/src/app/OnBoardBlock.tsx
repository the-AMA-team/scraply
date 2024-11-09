"use client";
import React from "react";

interface OnBoardBlockProps {
  label: string;
  color: string;
}

const OnBoardBlock = ({ label, color }: OnBoardBlockProps) => {
  return (
    <div
      className={`px-8 py-20 rounded-lg text-center ring-1 ring-zinc-100 mr-1`}
      style={{ backgroundColor: color }}
    >
      <div>{label}</div>
    </div>
  );
};

export default OnBoardBlock;
