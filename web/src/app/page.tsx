"use client";

import { useUser } from "@/contexts/UserContext";

interface PageProps {}

const Page = (props: PageProps) => {
  const { user, handleGoogleSignIn, handleGoogleSignOut } = useUser()!;

  return (
    <div className="text-white bg-zinc-900 h-screen">
      {/* {!user && (
        <button onClick={handleGoogleSignIn}>Sign in with Google</button>
      )}

      {user && <p> User: {user.displayName}</p>}

      {user && <button onClick={handleGoogleSignOut}>Log Out</button>} */}
      <div className="p-28 pt-72">
        <div className="font-bold text-6xl">
          Gamified Deep Learning Playground
        </div>
        <div className="w-2/3 text-zinc-300 pt-2">
          Something never seen before in education. DeepCraft takes DeepLearning
          and makes it simple to understand using a beautiful scratch-like user
          interface. Run custom deep learning models without any code as you try
          to lower your losses and increase your accuracy, with each epoch comes
          triumph!
        </div>
      </div>
      <img src="/gradient.png" alt="" className="bottom-0 absolute" />
    </div>
  );
};

export default Page;
