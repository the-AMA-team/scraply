"use client";
import React, { useState, useRef, useEffect } from "react";
import LayersTab from "./layers/LayersTab";
import { AppTabs } from "~/types/index";
import DATASETS from "~/util/DATASETS";
import Toggle from "../Toggle";
import TrainingTab from "./training/TrainingTab";
import OutputsTab from "./outputs/OutputsTab";
import { useBoardStore } from "~/state/boardStore";
import {
  generateUniqueBlocks,
  generateLeNetBlocks,
  generateResNetBlocks,
  getAvailableArchitectures,
} from "~/util/defaultConfigs";

const Board = () => {
  const [tab, setTab] = useState<AppTabs>(AppTabs.LAYERS);
  const [selectedDataset, setSelectedDataset] = useState<string>(
    DATASETS[0]!.inputName,
  );
  const [selectedArchitecture, setSelectedArchitecture] =
    useState<string>("custom");
  const { loadDefaultConfig, clearCanvas } = useBoardStore();
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setDropdownOpen(false);
      }
    };
    if (dropdownOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    } else {
      document.removeEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [dropdownOpen]);

  // Get available architectures for the current dataset
  const availableArchitectures = getAvailableArchitectures(selectedDataset);

  useEffect(() => {
    // If current architecture is not available for this dataset, reset to the first available one
    if (!availableArchitectures.includes(selectedArchitecture)) {
      setSelectedArchitecture(availableArchitectures[0] || "custom");
      return;
    }

    // When architecture changes, load or clear config
    if (selectedArchitecture === "custom") {
      clearCanvas();
    } else {
      const defaultBlocks =
        selectedArchitecture === "lenet"
          ? generateLeNetBlocks(selectedDataset)
          : selectedArchitecture === "resnet"
            ? generateResNetBlocks(selectedDataset)
            : generateUniqueBlocks(selectedDataset);
      loadDefaultConfig(defaultBlocks);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedArchitecture, selectedDataset, availableArchitectures]);

  const Tabs: Record<AppTabs, React.ReactNode> = {
    [AppTabs.LAYERS]: <LayersTab />,
    [AppTabs.TRAINING]: <TrainingTab selectedDataset={selectedDataset} />,
    [AppTabs.OUTPUTS]: <OutputsTab selectedDataset={selectedDataset} />,
  };

  return (
    <div className={`overflow-hidden bg-zinc-900 text-white`}>
      <div>
        <div className="flex justify-between p-4">
          <div className="mx-4 flex items-center space-x-6">
            <div className="flex items-center">
              <div className="mx-2 text-lg">Dataset</div>
              {/* Custom Dropdown */}
              <div className="relative min-w-[220px]" ref={dropdownRef}>
                <button
                  className="flex w-full items-center justify-between rounded bg-zinc-800 p-2 text-left text-white outline-none ring-1 ring-zinc-700 hover:bg-zinc-700"
                  onClick={() => setDropdownOpen((open) => !open)}
                  type="button"
                >
                  <span>
                    {DATASETS.find((d) => d.inputName === selectedDataset)
                      ?.label || "Select dataset"}
                  </span>
                  <svg
                    className={`ml-2 h-4 w-4 transition-transform ${
                      dropdownOpen ? "rotate-180" : ""
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 9l-7 7-7-7"
                    />
                  </svg>
                </button>
                {dropdownOpen && (
                  <div className="absolute z-50 mt-1 w-full rounded bg-zinc-800 shadow-lg ring-1 ring-zinc-700">
                    {["classification", "regression"].map((kind) => {
                      const group = DATASETS.filter((d) => d.kind === kind);
                      if (group.length === 0) return null;
                      return (
                        <div key={kind}>
                          <div className="px-3 py-1 text-xs font-semibold uppercase text-zinc-500">
                            {kind.charAt(0).toUpperCase() + kind.slice(1)}
                          </div>
                          {group.map((dataset) => (
                            <button
                              key={dataset.inputName}
                              className={`flex w-full flex-col items-start px-4 py-2 text-left transition-colors duration-75 hover:bg-zinc-700 focus:bg-zinc-700 ${
                                selectedDataset === dataset.inputName
                                  ? "bg-zinc-700 text-white"
                                  : "text-zinc-200"
                              }`}
                              onClick={() => {
                                setSelectedDataset(dataset.inputName);
                                setDropdownOpen(false);

                                // Check if current architecture is available for the new dataset
                                const newDatasetArchitectures =
                                  getAvailableArchitectures(dataset.inputName);
                                if (
                                  !newDatasetArchitectures.includes(
                                    selectedArchitecture,
                                  )
                                ) {
                                  setSelectedArchitecture(
                                    newDatasetArchitectures[0] || "custom",
                                  );
                                  clearCanvas();
                                  return;
                                }

                                if (selectedArchitecture === "custom") {
                                  clearCanvas();
                                } else {
                                  const defaultBlocks =
                                    selectedArchitecture === "lenet"
                                      ? generateLeNetBlocks(dataset.inputName)
                                      : selectedArchitecture === "resnet"
                                        ? generateResNetBlocks(
                                            dataset.inputName,
                                          )
                                        : generateUniqueBlocks(
                                            dataset.inputName,
                                          );
                                  loadDefaultConfig(defaultBlocks);
                                }
                              }}
                              type="button"
                            >
                              <span className="font-medium text-white">
                                {dataset.label}
                              </span>
                              <span className="text-xs text-zinc-400">
                                {dataset.summary}
                              </span>
                            </button>
                          ))}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
            <div className="flex items-center">
              <div className="mx-2 text-sm text-zinc-300">Architecture:</div>
              <select
                className="rounded bg-zinc-800 p-2 text-sm text-white outline-none ring-1 ring-zinc-700"
                value={selectedArchitecture}
                onChange={(e) => setSelectedArchitecture(e.target.value)}
              >
                {availableArchitectures.map((arch) => (
                  <option key={arch} value={arch}>
                    {arch.charAt(0).toUpperCase() + arch.slice(1)}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <Toggle
            color="blue"
            options={Object.values(AppTabs)}
            selected={tab}
            setSelected={setTab as React.Dispatch<React.SetStateAction<string>>}
          />

          {/* duplicate invisible component for centering, find a better way */}
          <div className="invisible mx-4 flex items-center space-x-6">
            <div className="flex items-center">
              <div className="mx-2 text-lg">Dataset</div>
              <select
                className="rounded bg-zinc-800 p-2 text-white outline-none"
                onChange={(e) => console.log(e.target.value)}
                value={selectedDataset}
              >
                {DATASETS.map((dataset, idx) => {
                  return (
                    <option key={idx} value={dataset.inputName}>
                      {dataset.label}
                    </option>
                  );
                })}
              </select>
            </div>
            <div className="flex items-center">
              <input
                type="checkbox"
                className="mr-2 h-4 w-4 rounded border-gray-300 bg-zinc-800 text-blue-600"
              />
              <label className="text-sm text-zinc-300">
                Load default model
              </label>
            </div>
          </div>
        </div>

        {Tabs[tab]}
      </div>
    </div>
  );
};

export default Board;
