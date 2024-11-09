"use client";
import React from "react";
import LEVEL_DESC from "../../../../levels/Desc";
import Board from "./Board";
import ReactConfetti from "react-confetti";

interface LevelProps {
  level: number;
}

const Level = ({ level }: LevelProps) => {
  return (
    <div className="h-screen text-white bg-zinc-900">
      <ReactConfetti
        run={true}
        height={window.screen.height - 125}
        width={window.screen.width - 100}
      />

      <div className="pt-10 px-16">
        <div className="text-3xl">Level {level}</div>
        <div className="text-xl">{LEVEL_DESC[level - 1].title}</div>
        <div className="text-md w-1/2 text-zinc-300">
          {LEVEL_DESC[level - 1].prompt}
        </div>
      </div>
      <Board />
    </div>
  );
};

export default Level;
