"use client";
import React from "react";

interface OnBoardBlockProps {
  label: string;
  color: string;
}

const OnBoardBlock = ({ label, color }: OnBoardBlockProps) => {
  return (
    <div
      className={`pl-8 py-20 rounded-2xl text-center ring-1 ring-zinc-100 mr-1 flex`}
      style={{ backgroundColor: color }}
    >
      <div className="relative overflow-visible">
        <div className="my-auto">{label}</div>
        <div>
          <select
            name="cars"
            id="cars"
            className="h-16 w-16 rounded-full text-zinc-900 outline-none absolute right-[-20px]"
          >
            <option value="volvo">Volvo</option>
            <option value="saab">Saab</option>
            <option value="mercedes">Mercedes</option>
            <option value="audi">Audi</option>
          </select>
        </div>
      </div>
    </div>
  );
};

export default OnBoardBlock;
