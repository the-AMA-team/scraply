import { TrainingResult } from "~/types/index";
import { ResponsiveLine } from "@nivo/line";
import { useTrainingStore } from "~/state/trainingStore";
import { useDownloadFile } from "~/hooks/useApi";
import ModelMiniMap from "./ModelMiniMap";

interface HistoryItemProps {
  idx: number;
  trainingRes: TrainingResult;
  nextTrainingRes?: TrainingResult;
}

const HistoryItem: React.FC<HistoryItemProps> = ({
  idx,
  trainingRes,
  nextTrainingRes,
}) => {
  const getDiffUI = (diff: number) => {
    const roundedDiff = Math.round(diff);
    if (roundedDiff === 0) return null;
    if (roundedDiff > 0) {
      return <span className="text-sm text-green-400">+{roundedDiff}%</span>;
    } else if (roundedDiff < 0) {
      return <span className="text-sm text-red-400">{roundedDiff}%</span>;
    }
  };

  const { openHistoryItemIdx, setOpenHistoryItem } = useTrainingStore();
  const { mutate: downloadFile } = useDownloadFile();
  const lossPoints = (trainingRes.train_losses ?? []).filter(
    (p) => Number.isFinite(p.x) && Number.isFinite(p.y),
  );

  return (
    <div className="group my-3 overflow-hidden rounded-xl border border-slate-700/50 bg-zinc-900 shadow-lg backdrop-blur-sm transition-all duration-300 hover:shadow-xl">
      {/* Header */}
      <div className="px-5 pb-2 pt-4">
        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={() =>
              setOpenHistoryItem(idx === openHistoryItemIdx ? null : idx)
            }
            className="flex-1 text-left"
          >
            <h3 className="text-lg font-bold text-slate-100">
              {trainingRes.run_name || `Training Run ${idx}`}
            </h3>
            <p className="text-sm text-slate-400">Run #{idx}</p>
          </button>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => downloadFile(trainingRes.trainingConfig)}
              className="flex items-center gap-2 rounded-lg bg-zinc-700 px-3 py-1.5 text-sm text-zinc-200 transition-colors duration-200 hover:bg-zinc-600"
              title="Download Python Notebook"
              aria-label="Download Python Notebook"
            >
              <svg
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              <span className="text-sm">Python Notebook</span>
            </button>
            <button
              type="button"
              onClick={() =>
                setOpenHistoryItem(idx === openHistoryItemIdx ? null : idx)
              }
              aria-label={
                openHistoryItemIdx === idx ? "Collapse run" : "Expand run"
              }
            >
              <svg
                className={`h-5 w-5 text-slate-400 transition-transform duration-200 ${
                  openHistoryItemIdx === idx ? "rotate-180" : ""
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
          </div>
        </div>
      </div>

      {/* Summary Metrics */}
      <div className="px-6 pb-4">
        <div className="space-y-4">
          {/* Train Accuracy */}
          <div className="rounded-xl border border-white/10 bg-white/5 p-4 backdrop-blur-sm transition-all duration-200">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-sm font-medium text-slate-300">
                Train Accuracy
              </span>
              <div className="flex items-center gap-3">
                <span className="text-xl font-bold text-slate-100">
                  {Math.round(trainingRes.avg_train_acc)}%
                </span>
                {nextTrainingRes && (
                  <div className="min-w-[60px] text-right">
                    {getDiffUI(
                      trainingRes.avg_train_acc - nextTrainingRes.avg_train_acc,
                    )}
                  </div>
                )}
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-slate-700/50 shadow-inner">
                <div
                  className="h-full rounded-full bg-emerald-500 shadow-sm transition-all duration-500"
                  style={{
                    width: `${Math.min(100, Math.max(0, trainingRes.avg_train_acc))}%`,
                  }}
                />
              </div>
              {nextTrainingRes && (
                <div className="min-w-[60px]">
                  {/* Spacer for diff alignment */}
                </div>
              )}
            </div>
          </div>

          {/* Test Accuracy */}
          <div className="rounded-xl border border-white/10 bg-white/5 p-4 backdrop-blur-sm transition-all duration-200">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-sm font-medium text-slate-300">
                Test Accuracy
              </span>
              <div className="flex items-center gap-3">
                <span className="text-xl font-bold text-slate-100">
                  {Math.round(trainingRes.avg_test_acc)}%
                </span>
                {nextTrainingRes && (
                  <div className="min-w-[60px] text-right">
                    {getDiffUI(
                      trainingRes.avg_test_acc - nextTrainingRes.avg_test_acc,
                    )}
                  </div>
                )}
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-slate-700/50 shadow-inner">
                <div
                  className="h-full rounded-full bg-zinc-600 shadow-sm transition-all duration-500"
                  style={{
                    width: `${Math.min(100, Math.max(0, trainingRes.avg_test_acc))}%`,
                  }}
                />
              </div>
              {nextTrainingRes && (
                <div className="min-w-[60px]">
                  {/* Spacer for diff alignment */}
                </div>
              )}
            </div>
          </div>

          {/* Train Loss */}
          <div className="rounded-xl border border-white/10 bg-white/5 p-4 backdrop-blur-sm transition-all duration-200">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-sm font-medium text-slate-300">
                Train Loss
              </span>
              <div className="flex items-center gap-3">
                <span className="font-mono text-lg font-semibold text-slate-100">
                  {trainingRes.avg_train_loss.toFixed(4)}
                </span>
                {nextTrainingRes && (
                  <div className="min-w-[60px] text-right">
                    {getDiffUI(
                      -(
                        trainingRes.avg_train_loss -
                        nextTrainingRes.avg_train_loss
                      ) * 100,
                    )}
                  </div>
                )}
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-slate-700/50 shadow-inner">
                <div
                  className="h-full rounded-full bg-zinc-600 shadow-sm transition-all duration-500"
                  style={{
                    width: `${Math.max(10, 100 - trainingRes.avg_train_loss * 20)}%`,
                  }}
                />
              </div>
              {nextTrainingRes && (
                <div className="min-w-[60px]">
                  {/* Spacer for diff alignment */}
                </div>
              )}
            </div>
          </div>

          {/* Test Loss */}
          <div className="rounded-xl border border-white/10 bg-white/5 p-4 backdrop-blur-sm transition-all duration-200">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-sm font-medium text-slate-300">
                Test Loss
              </span>
              <div className="flex items-center gap-3">
                <span className="font-mono text-lg font-semibold text-slate-100">
                  {trainingRes.avg_test_loss.toFixed(4)}
                </span>
                {nextTrainingRes && (
                  <div className="min-w-[60px] text-right">
                    {getDiffUI(
                      -(
                        trainingRes.avg_test_loss -
                        nextTrainingRes.avg_test_loss
                      ) * 100,
                    )}
                  </div>
                )}
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-slate-700/50 shadow-inner">
                <div
                  className="h-full rounded-full bg-zinc-600 shadow-sm transition-all duration-500"
                  style={{
                    width: `${Math.max(10, 100 - trainingRes.avg_test_loss * 20)}%`,
                  }}
                />
              </div>
              {nextTrainingRes && (
                <div className="min-w-[60px]">
                  {/* Spacer for diff alignment */}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Collapsible Details */}
      {openHistoryItemIdx === idx && (
        <div className="space-y-5 p-4">
          {/* Loss Graph Section */}
          <div>
            <h3 className="mb-3 text-lg font-medium text-zinc-200">
              Loss Graph
            </h3>
            <div className="h-80 rounded-lg bg-zinc-900/50 p-3 ring-1 ring-zinc-700/30">
              {lossPoints.length >= 2 ? (
              <ResponsiveLine
                data={[
                  {
                    id: "train_loss",
                    data: lossPoints,
                  },
                ]}
                margin={{ top: 20, right: 20, bottom: 50, left: 60 }}
                enableGridX={false}
                enableGridY={true}
                gridYValues={5}
                xScale={{ type: "linear", min: 0, max: "auto" }}
                yScale={{
                  type: "linear",
                  min: "auto",
                  max: "auto",
                  stacked: false,
                  reverse: false,
                }}
                colors={["#10b981"]}
                theme={{
                  background: "transparent",
                  text: {
                    fontSize: 12,
                    fill: "#a1a1aa",
                    outlineWidth: 0,
                    outlineColor: "transparent",
                  },
                  tooltip: {
                    container: {
                      background: "#27272a",
                      color: "#e4e4e7",
                      fontSize: 12,
                      borderRadius: "6px",
                      boxShadow:
                        "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
                      border: "1px solid #3f3f46",
                    },
                  },
                  axis: {
                    domain: {
                      line: {
                        stroke: "#52525b",
                        strokeWidth: 1,
                      },
                    },
                    legend: {
                      text: {
                        fontSize: 12,
                        fill: "#e4e4e7",
                      },
                    },
                    ticks: {
                      line: {
                        stroke: "#52525b",
                        strokeWidth: 1,
                      },
                      text: {
                        fontSize: 11,
                        fill: "#a1a1aa",
                      },
                    },
                  },
                  grid: {
                    line: {
                      stroke: "#3f3f46",
                      strokeWidth: 1,
                      strokeOpacity: 0.3,
                    },
                  },
                  crosshair: {
                    line: {
                      stroke: "#a1a1aa",
                      strokeWidth: 1,
                      strokeOpacity: 0.75,
                    },
                  },
                }}
                axisTop={null}
                axisRight={null}
                axisBottom={{
                  tickSize: 5,
                  tickPadding: 5,
                  tickRotation: 0,
                  legend: "Epoch",
                  legendOffset: 36,
                  legendPosition: "middle",
                  tickValues:
                    lossPoints.length > 20
                      ? Array.from(
                          {
                            length: Math.min(
                              10,
                              lossPoints.length,
                            ),
                          },
                          (_, i) =>
                            Math.floor(
                              (i * (lossPoints.length - 1)) /
                                (Math.min(10, lossPoints.length) -
                                  1),
                            ),
                        )
                      : undefined,
                }}
                axisLeft={{
                  tickSize: 5,
                  tickPadding: 5,
                  tickRotation: 0,
                  legend: "Loss",
                  legendOffset: -50,
                  legendPosition: "middle",
                }}
                pointSize={4}
                pointColor="#10b981"
                pointBorderWidth={2}
                pointBorderColor="#ffffff"
                pointLabelYOffset={-12}
                useMesh={true}
                curve={lossPoints.length >= 3 ? "monotoneX" : "linear"}
                lineWidth={2}
                enableArea={true}
                areaOpacity={0.1}
                legends={[]}
                animate={false}
              />
              ) : (
                <div className="flex h-full items-center justify-center text-sm text-zinc-500">
                  Not enough loss data to graph
                </div>
              )}
            </div>
          </div>

          {/* Model Config Section */}
          <div>
            <h3 className="mb-3 text-lg font-medium text-zinc-200">
              Model Configuration
            </h3>
            <div className="rounded-lg bg-zinc-900/50 p-4 ring-1 ring-zinc-700/30">
              <ModelMiniMap trainingConfig={trainingRes.trainingConfig} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default HistoryItem;
