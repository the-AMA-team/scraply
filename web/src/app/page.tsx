"use client";
import Board from "./board/Board";
import { useState } from "react";
import { auth, provider } from "../../utils/firebase";
import { signInWithPopup, signOut } from "firebase/auth";
import { POST } from "./api/user/route";
import { NextRequest } from "next/server";
import axios from "axios";

interface PageProps {}

const Page = (props: PageProps) => {

  const [user, setUser] = useState<any>(null);

  const handleGoogleSignIn = async () => {
    try {
      const result = await signInWithPopup(auth, provider);
      const user = result.user;
      setUser(user);
      console.log("User Info: ", user);
      // await POST(new NextRequest({id : user.uid, name : user.displayName}))
      const data = {
        id: user.uid,
        name: user.displayName
      }

      try {
        const response = await axios.post("http://localhost:3000/api/user", data);
        console.log(response);
      } catch (error) {
        console.error(error);
      }

      
    } catch (error: any) {
      console.error("Error during Google sign-in:", error.message);
    }
  };

  const handleGoogleSignOut = async () => {
    try {
      await signOut(auth);
      setUser(null);
      console.log("User signed out successfully");
    } catch (error) {
      console.error("Error signing out: ", error);
    }
  }

  return <div className="text-white bg-zinc-900">
    {
    !user &&
    <button onClick={handleGoogleSignIn}>
      Sign in with Google
    </button>
    }

    {
      user && <p> User: { user.displayName }</p>
    }

    {
      user && <button onClick={handleGoogleSignOut}>
        Log Out
      </button>
    }
  </div>;
};

export default Page;
