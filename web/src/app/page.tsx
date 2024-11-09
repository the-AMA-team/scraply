"use client";

import { useUser } from "@/contexts/UserContext";

interface PageProps {}

const Page = (props: PageProps) => {
  const { user, handleGoogleSignIn, handleGoogleSignOut } = useUser()!;

  return (
    <div className="text-white bg-zinc-900 h-screen">
      {!user && (
        <button onClick={handleGoogleSignIn}>Sign in with Google</button>
      )}

      {user && <p> User: {user.displayName}</p>}

      {user && <button onClick={handleGoogleSignOut}>Log Out</button>}
    </div>
  );
};

export default Page;
