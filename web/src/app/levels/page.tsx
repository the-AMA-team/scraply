import React from "react";
import LEVEL_DESC from "../../../../levels/Desc";

const page = () => {
  return (
    <div className="flex bg-zinc-100 h-screen">
      {[0, 1, 2, 3].map((idx: number) => {
        return (
          <button
            key={idx}
            className="h-52 w-52 bg-red-500 m-4 rounded-lg text-white text-center drop-shadow-xl"
          >
            <div className="">Level {idx + 1}</div>
            {/* <h1>Level {LEVEL_DESC[idx].title}</h1>
            <p>Level {LEVEL_DESC[idx].prompt} is the best level</p> */}
          </button>
        );
      })}
    </div>
  );
};

export default page;
