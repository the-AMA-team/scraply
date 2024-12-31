"use client";

import { redirect } from "next/navigation";
import { useEffect } from "react";

interface PageProps {}

const Page = (props: PageProps) => {
  return (
    <div className="relative text-white bg-zinc-900 h-screen text-center text-3xl font-bold">
      Scraply
    </div>
  );

  //   return (
  //     <div>
  //       {user && <p> User: {user.displayName}</p>}
  //       {user && <button onClick={handleGoogleSignOut}>Log Out</button>}
  //     </div>
  // );
};

export default Page;
