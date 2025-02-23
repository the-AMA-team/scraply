"use client";
import { useUser } from "@clerk/nextjs";
import Link from "next/link";

const Landing = () => {
  const user = useUser();
  console.log(user);
  return (
    <div className="flex justify-between">
      <div className="h-screen w-2/3 bg-white px-56 text-zinc-900">
        <div className="text-center text-4xl">rawr</div>
      </div>
      <div className="flex h-screen items-center space-x-4">
        <button className="flex justify-center">
          <Link href="/teachers" className="rounded-2xl bg-blue-600 px-8 py-4">
            {"I'm a Teacher ->"}
          </Link>
        </button>
        <button className="flex justify-center">
          <Link href="/teachers" className="rounded-2xl bg-teal-700 px-8 py-4">
            {"I'm a Student ->"}
          </Link>
        </button>
      </div>
    </div>
  );
};

export default Landing;
