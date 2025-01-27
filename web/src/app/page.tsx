"use client";
import Board from "./board/Board";
import "../dynamic-model-js/main"

export default function HomePage() {
  return (
    <div className="h-screen overflow-hidden bg-zinc-900 text-white">
      <Board />
    </div>
  );
}
