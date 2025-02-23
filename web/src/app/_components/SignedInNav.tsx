import { SignedIn, UserButton } from "@clerk/nextjs";

const SignedInNav = () => {
  return (
    <div className="flex justify-between bg-zinc-800 text-white">
      <div className="flex">
        <img src="favicon.png" className="mx-4 w-12" alt="" />
        <div className="py-4 pr-7">rawr</div>
      </div>
      <div className="my-auto px-4">
        <UserButton />
      </div>
    </div>
  );
};

export default SignedInNav;
