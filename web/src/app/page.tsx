"use client";
import Board from "./board/Board";
// import "../dynamic-model-js/main" // just for testing; will run on the server

export default function HomePage() {
  return (
    <div className="h-full bg-zinc-900 text-white">
      <Board />
    </div>
  );
}
