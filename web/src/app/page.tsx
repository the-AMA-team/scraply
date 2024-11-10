"use client";

import { useUser } from "@/contexts/UserContext";
import { redirect } from "next/navigation";
import { useEffect } from "react";

interface PageProps {}

const Page = (props: PageProps) => {
  const { user, handleGoogleSignIn, handleGoogleSignOut } = useUser()!;
  useEffect(() => {
    if (user) {
      redirect("/board/1");
    }
  }, [user]);

  return (
    <div className="relative text-white bg-zinc-900 h-screen">
      <img
        src="/gradient.png"
        alt=""
        className="absolute inset-0 w-full h-full object-cover z-0"
      />
      <div className="relative z-10 p-28 pt-72">
        <div className="font-bold text-6xl">Scraply</div>
        <div className="w-2/3 text-zinc-300 pt-2">
          Something never seen before in education. DeepCraft takes DeepLearning
          and makes it simple to understand using a beautiful scratch-like user
          interface. Run custom deep learning models without any code as you try
          to lower your losses and increase your accuracy, with each epoch comes
          triumph!
        </div>
        <button
          onClick={handleGoogleSignIn}
          className={`my-4 text-lg px-4 py-2 rounded-2xl transition-all ease-in-out bg-blue-500 ring-indigo-500 duration-300 hover:bg-indigo-600 hover:ring-2 active:bg-indigo-500`}
        >
          Sign in with Google
        </button>
      </div>
    </div>
  );

  // return (
  //   <div>
  //     {user && <p> User: {user.displayName}</p>}
  //     {user && <button onClick={handleGoogleSignOut}>Log Out</button>}
  //   </div>
  // );
};

export default Page;
