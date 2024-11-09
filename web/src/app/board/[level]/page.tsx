import React from "react";
import LEVEL_DESC from "../../../../../levels/Desc";
import Board from "../Board";

const page = async ({ params }: { params: Promise<{ level: string }> }) => {
  const level = parseInt((await params).level);

  if (level < 1 || level > 4) {
    return (
      <div className="h-screen text-white bg-zinc-900">
        <div className="text-3xl">Level doesn't exist</div>
      </div>
    );
  }

  return (
    <div className="h-screen text-white bg-zinc-900">
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

export default page;
