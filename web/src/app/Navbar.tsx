"use client";
import { SignedOut, SignInButton, SignedIn, UserButton } from "@clerk/nextjs";
import { useDemo } from "~/state/DemoContext";

const Navbar = () => {
  const { isDemoing, setIsDemoing } = useDemo();
  return (
    <>
      <SignedOut>
        <div className="flex justify-between bg-zinc-800 text-white">
          <div className="flex">
            <img src="favicon.png" className="my-auto ml-6 h-8" alt="" />
            <div className="mx-4 py-4 pr-7 font-semibold">scraply</div>
          </div>

          <SignInButton>
            <button className="mx-2 my-2 rounded-lg bg-blue-600 px-6 py-2">
              Sign In
            </button>
          </SignInButton>
        </div>
      </SignedOut>
      <SignedIn>
        <div className="flex justify-between bg-zinc-800 text-white">
          <div className="flex">
            <img src="favicon.png" className="my-auto ml-4 h-8" alt="" />
            <div className="mx-4 py-4 pr-7 font-semibold">scraply</div>
          </div>
          <div className="flex">
            <div className="my-auto px-2">
              <UserButton />
            </div>
            <button
              className="relative mx-4 my-auto h-4/5 rounded-md bg-blue-500 px-4 hover:bg-blue-600"
              onClick={() => setIsDemoing(true)}
            >
              Demo
            </button>
          </div>
        </div>
      </SignedIn>
    </>
  );
};

export default Navbar;
