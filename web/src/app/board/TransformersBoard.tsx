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
    <div className="my-4">
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
        <div className="my-1 flex flex-col items-center rounded-xl bg-zinc-900 p-1">
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
        <div className="my-1 flex flex-col items-center rounded-xl bg-zinc-900 p-1">
          <div>Layer Norm</div>
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
    <div>
      <div className="full flex justify-center">
        <div className="w-1/2">
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
      </div>
      <div className="flex w-full justify-center">
        <button
          onClick={handleAppendDecoder}
          className="m-2 w-1/2 rounded-2xl border-2 border-dashed border-zinc-800 p-2 text-3xl text-zinc-800 transition-colors duration-75 hover:border-zinc-700 hover:text-zinc-700"
        >
          +
        </button>
      </div>
    </div>
  );
};

export default TransformersBoard;
