"use client";
import { useUser } from "@clerk/nextjs";
import Board from "./board/Board";

const Landing = () => {
  const user = useUser();
  console.log(user);
  return <Board />;
};

export default Landing;
