"use client";
import Board from "./Board";

interface PageProps {}

const Page = (props: PageProps) => {
  return (
    <div className="text-white bg-zinc-900">
      <Board />
    </div>
  );
};

export default Page;
