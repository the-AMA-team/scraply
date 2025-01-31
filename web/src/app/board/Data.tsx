import React, { useState } from "react";
import Toggle from "../_components/Toggle";
import { DataMode } from "~/types";

interface DataProps {
  dataModeState: [DataMode, React.Dispatch<React.SetStateAction<DataMode>>];
}

const Data = ({ dataModeState }: DataProps) => {
  const [dataMode, setDataMode] = dataModeState;

  // const Upload = () => {
  //   return (
  //     <div>
  //       <div>Upload your own dataset!</div>
  //       <div>Coming to Scraply soon!</div>
  //     </div>
  //   );
  // };

  const Preset = () => {
    return (
      <div>
        <div>Select Preset</div>
        <div className="flex justify-center">
          <select className="w-1/2 rounded-lg bg-zinc-700 p-2 outline-none ring-1 ring-zinc-600">
            <option value="mnist">MNIST</option>
          </select>
        </div>
      </div>
    );
  };

  return (
    <div className="h-screen">
      <div className="mx-16 my-10">
        <div className="">Select Data</div>
        <div className="my-6 rounded-xl p-4 ring-2 ring-zinc-700">
          {/* <Toggle
            option1="UPLOAD"
            option2="PRESET"
            color="zinc"
            selected={dataMode}
            setSelected={
              setDataMode as React.Dispatch<React.SetStateAction<string>>
            }
          /> */}
          <div className="flex justify-between">
            <div className="w-1/2">
              {/* <div>{dataMode === "UPLOAD" ? <Upload /> : <Preset />}</div> */}
              <Preset />
            </div>
            <div className="w-1/2 rounded-lg bg-zinc-700 p-2 ring-1 ring-zinc-600">
              Pre-processing
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Data;
