import { useState } from "react";

interface Decoder {
  title: string;
  ffLinearLayers: number;
  saEmbeddingDim: number;
  saAttentionHeads: number;
}

interface DecoderProps extends Decoder {
  setFfLinearLayers: (value: number) => void;
  setSaEmbeddingDim: (value: number) => void;
  setSaAttentionHeads: (value: number) => void;
}

const Decoder: React.FC<DecoderProps> = ({
  title,
  ffLinearLayers,
  saEmbeddingDim,
  saAttentionHeads,
  setFfLinearLayers,
  setSaEmbeddingDim,
  setSaAttentionHeads,
}) => {
  return (
    <div className="my-4 w-2/3">
      <div className="rounded-xl bg-zinc-900 p-2 text-center text-2xl text-zinc-700">
        {title}
      </div>
      <div className="rounded-3xl p-2 ring ring-zinc-800">
        <div className="my-1 flex flex-col items-center rounded-2xl bg-zinc-800 p-3">
          <div>Feed Forward</div>
          <div className="flex items-center">
            <div>Linear layers: </div>
            <input
              type="number"
              className="mx-2 w-10 rounded-md text-center text-zinc-900"
              value={ffLinearLayers}
              onChange={(e) => setFfLinearLayers(parseInt(e.target.value))}
            />
          </div>
        </div>
        <div className="my-1 flex flex-col items-center rounded-xl bg-zinc-900 p-1 ring-1 ring-zinc-800">
          <div>Layer Norm</div>
        </div>
        <div className="my-1 flex flex-col items-center rounded-2xl bg-zinc-800 p-3">
          <div>Self Attention</div>
          <div className="flex items-center">
            <div>Embedding Dim: </div>
            <input
              type="number"
              className="mx-2 w-10 rounded-md text-center text-zinc-900"
              value={saEmbeddingDim}
              onChange={(e) => setSaEmbeddingDim(parseInt(e.target.value))}
            />
            <div>Attention Heads: </div>
            <input
              type="number"
              className="mx-2 w-10 rounded-md text-center text-zinc-900"
              value={saAttentionHeads}
              onChange={(e) => setSaAttentionHeads(parseInt(e.target.value))}
            />
          </div>
        </div>
        <div className="my-1 flex flex-col items-center rounded-xl bg-zinc-900 p-1 ring-1 ring-zinc-800">
          <div>Layer Norm</div>
        </div>
      </div>
    </div>
  );
};

const TrainConfig = () => {
  const [loss, setLoss] = useState("BCE");
  const [optimizer, setOptimizer] = useState("Adam");
  const [learningRate, setLearningRate] = useState(0.001);
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(10);

  const [isTraining, setIsTraining] = useState(false);
  const [trainingRes, setTrainingRes] = useState<any | null>(null);
  const [progress, setProgress] = useState(0);

  return (
    <div className="my-4 w-fit">
      <div className="bg-zinc-900 p-2 text-center text-2xl text-zinc-700">
        Train
      </div>
      <div className="rounded-lg bg-zinc-800 p-1 px-2 py-1 text-sm">
        <div>
          <div className="my-1 flex">
            Loss:{" "}
            <select
              className="mx-1 cursor-pointer rounded bg-zinc-700 p-1 text-sm text-white outline-none"
              value={loss}
              onChange={(e) => setLoss(e.target.value)}
            >
              <option value="BCE">BCE</option>
              <option value="CrossEntropy">CrossEntropy</option>
            </select>
          </div>
          <div className="my-1 flex">
            Optimizer:{" "}
            <select
              className="mx-1 cursor-pointer rounded bg-zinc-700 p-1 text-sm text-white outline-none"
              value={optimizer}
              onChange={(e) => setOptimizer(e.target.value)}
            >
              <option value="Adam">Adam</option>
              <option value="AdamW">AdamW</option>
              <option value="SGD">SGD</option>
              <option value="RMSprop">RMSprop</option>
            </select>
          </div>
          <div className="my-1 flex">
            Learning Rate:{" "}
            <input
              type="range"
              name="Learning Rate"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
              min={0.001}
              max={0.1}
              step={0.001}
            />
            <input
              type="number"
              className="mx-1 w-14 rounded bg-zinc-700 py-1 text-right outline-none"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            />
          </div>
          <div className="my-1 flex">
            Epochs:{" "}
            <input
              type="range"
              name="Batch Size"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value))}
              min={1}
              max={1000}
            />
            <input
              type="number"
              className="mx-1 w-14 rounded bg-zinc-700 py-1 text-right outline-none"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value))}
            />
          </div>
          <div className="my-1 flex">
            Batch Size:{" "}
            <input
              type="range"
              name="Batch Size"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value))}
              min={1}
              max={100}
            />
            <input
              type="number"
              className="mx-1 w-14 rounded bg-zinc-700 py-1 text-right outline-none"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value))}
            />
          </div>
          <div className="m-2 flex justify-center">
            <div className="flex rounded-2xl bg-blue-500 px-4 py-2 text-white">
              <button
                disabled={isTraining}
                className={`text-lg transition-all ease-in-out ${
                  !isTraining &&
                  "hover:bg-indigo-600 hover:ring-2 active:bg-indigo-500"
                } ring-indigo-500 duration-300 ${
                  isTraining && "animate-pulse"
                }`}
              >
                {isTraining ? "Training..." : "Train"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const TransformersBoard = () => {
  const [decoders, setDecoders] = useState<Decoder[]>([]);
  const handleAppendDecoder = () => {
    setDecoders((prev) => [
      ...prev,
      {
        title: `Decoder #${prev.length + 1}`,
        ffLinearLayers: 3,
        saEmbeddingDim: 64,
        saAttentionHeads: 6,
      },
    ]);
  };
  return (
    <div className="flex">
      <div className="w-2/3">
        <div className="flex flex-col items-center">
          {decoders.map((encoder, i) => (
            <Decoder
              key={i}
              title={encoder.title}
              setFfLinearLayers={(value) =>
                setDecoders((prev) => {
                  const newDecoder = [...prev];
                  newDecoder[i].ffLinearLayers = value;
                  return newDecoder;
                })
              }
              setSaEmbeddingDim={(value) =>
                setDecoders((prev) => {
                  const newDecoder = [...prev];
                  newDecoder[i].saEmbeddingDim = value;
                  return newDecoder;
                })
              }
              setSaAttentionHeads={(value) =>
                setDecoders((prev) => {
                  const newDecoder = [...prev];
                  newDecoder[i].saAttentionHeads = value;
                  return newDecoder;
                })
              }
              ffLinearLayers={encoder.ffLinearLayers}
              saEmbeddingDim={encoder.saEmbeddingDim}
              saAttentionHeads={encoder.saAttentionHeads}
            />
          ))}
        </div>
        <div className="flex justify-center">
          <button
            onClick={handleAppendDecoder}
            className="m-2 w-2/3 rounded-2xl border-2 border-dashed border-zinc-800 p-2 text-3xl text-zinc-800 transition-colors duration-75 hover:border-zinc-700 hover:text-zinc-700"
          >
            +
          </button>
        </div>
      </div>
      <div className="w-1/3">
        <TrainConfig />
      </div>
    </div>
  );
};

export default TransformersBoard;
