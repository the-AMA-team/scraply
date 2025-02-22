import React from "react";

interface ToggleProps {
  option1: string;
  option2: string;
  color: string;
  selected: string;
  setSelected: React.Dispatch<React.SetStateAction<string>>;
}

function Toggle({
  option1,
  option2,
  color,
  selected,
  setSelected,
}: ToggleProps) {
  return (
    <div
      className={`flex w-min rounded-xl bg-zinc-800 p-1 ring-1 ring-${color}-600`}
    >
      {[option1, option2].map((option) => (
        <div
          key={option}
          className={`cursor-pointer rounded-lg px-6 py-1 capitalize ${
            selected === option && `bg-${color}-600 text-white`
          }`}
          onClick={() => {
            setSelected(option);
          }}
        >
          {/* Title case */}
          {String(option).charAt(0).toUpperCase() +
            String(option).slice(1).toLowerCase()}
        </div>
      ))}
    </div>
  );
}

export default Toggle;
