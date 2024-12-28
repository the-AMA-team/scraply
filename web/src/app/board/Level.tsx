"use client";
import React, { useEffect, useState } from "react";
import LEVEL_DESC from "../../../../levels/Desc";
import Board from "./Board";
import ReactConfetti from "react-confetti";
import { useUser } from "@/contexts/UserContext";

interface LevelProps {
  level: number;
}

const Level = ({ level }: LevelProps) => {
  const { user, handleGoogleSignOut } = useUser()!;
  const [runConfetti, setRunConfetti] = useState(false);
  const [showConfetti, setShowConfetti] = useState(false);

  useEffect(() => {
    if (showConfetti) {
      setRunConfetti(true);
      setTimeout(() => {
        setShowConfetti(false);
      }, 2000);
    }
  }, [showConfetti]);

  return (
    <div className="h-screen text-white bg-zinc-900 overflow-hidden">
      <ReactConfetti
        run={runConfetti}
        recycle={showConfetti}
        height={window.screen.height - 125}
        width={window.screen.width - 100}
      />

      <div className="pt-10 px-16">
        {user && (
          <div className="flex justify-end">
            {
              <button onClick={handleGoogleSignOut} className="mx-2">
                Log Out {user.displayName}
              </button>
            }
          </div>
        )}
        <div className="text-3xl">Level {level}</div>
        <div className="text-xl">{LEVEL_DESC[level - 1].title}</div>
        <div className="text-md w-1/2 text-zinc-300">
          {LEVEL_DESC[level - 1].prompt}
        </div>
      </div>
      <Board setShowConfetti={setShowConfetti} level={level} />
    </div>
  );
};

export default Level;
