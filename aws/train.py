import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import boto3
import argparse
import os
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import s3fs

"""# dataset preparation"""

# Dataset class
class TextDataset(Dataset): # using pytorch dataset class
    def __init__(self, samples, word_to_int):
        self.samples = samples # storing samples
        self.word_to_int = word_to_int # storing word (to) indexes
    def __len__(self):
        return len(self.samples) # number of samples
    def __getitem__(self, idx):
        sample = self.samples[idx] # retrieving ith sample
        input_seq = torch.LongTensor([self.word_to_int[word] for word in sample[:-1]]) # input
        target_seq = torch.LongTensor([self.word_to_int[word] for word in sample[1:]]) # target words (slides over by 1 each time)
        # remember --> only one target is being outputted each time!
        return input_seq, target_seq

#CAUsAL MASKING
def generate_square_subsequent_mask(sz):
  """
  Generate a square mask for the sequence. The masked positions are filled with float('-inf').
  Unmasked positions are filled with float(0.0).
  """
  mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
  mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
  return mask

"""# Transformer model"""

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(max_len=SEQUENCE_LENGTH, d_model=embed_dim)
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )  # maybe in scraply --> would have to make a list of decoder_layer specifications. and use for loop in forward functions to get through each layer configuration
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=num_layers, # number of decoders stacked together (number of instances of decoders stacked together)
        )
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(0.3) # originally 0.2

    # Positional encoding is required. Else the model does not learn.
    def forward(self, x):
        emb = self.emb(x)

        # Generate input sequence mask with shape (SEQUENCE_LENGTH, SEQUENCE_LENGTH)
        input_mask = generate_square_subsequent_mask(x.size(1)).to(x.device)

        x = self.pos_encoder(emb)
        x = self.decoder(x, memory=x, tgt_mask=input_mask, memory_mask=input_mask)
        # can do -->
        # x = self.decoder_layer(x, memory=x, tgt_mask=input_mask, memory_mask=input_mask) INSTEAD
        x = self.dropout(x)
        out = self.linear(x)
        return out

# POSITIONAL ENCODING
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        """
        :param max_len: Input length sequence.
        :param d_model: Embedding dimension.
        :param dropout: Dropout value (default=0.1)
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Inputs of forward function
        :param x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[:, :x.size(1)]  # first generate positional encodings
        return self.dropout(x) # do some dropout i guess
def train(model, epochs, dataloader, criterion):
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for input_seq, target_seq in dataloader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            outputs = model(input_seq)
            target_seq = target_seq.contiguous().view(-1)
            outputs = outputs.view(-1, vocab_size)
            loss = criterion(outputs, target_seq.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().cpu().numpy()
        print(f"Epoch {epoch + 1}/{epochs} Loss: {running_loss / len(dataloader):.3f}")

if __name__ == "__main__":

    # Argument parsing for SageMaker
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    args, _ = parser.parse_known_args()
    # ... other arguments ...

    # Load data from the local SageMaker channel
    data_path = os.path.join(args.train, 'alice_1.txt')  # File path in the 'train' channel
    # data_path = 'training-data/alice_1.txt'  # Local file path
    with open(data_path, 'r') as file:
        text = file.read()

    print(text[:1000])

    # Tokenization
    words = text.split()
    word_counts = Counter(words)
    vocab = list(word_counts.keys())
    vocab_size = len(vocab)
    word_to_int = {word: i for i, word in enumerate(vocab)}
    int_to_word = {i: word for word, i in word_to_int.items()}

    print("Done tokenizing")

    SEQUENCE_LENGTH = 64
    samples = [words[i:i+SEQUENCE_LENGTH+1] for i in range(len(words)-SEQUENCE_LENGTH)]

    dataset = TextDataset(samples, word_to_int)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print("Done creating dataset")

    """# causal masking and positional encoding"""

    """# Training"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(vocab_size, embed_dim=100, num_layers=2, num_heads=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Starting training")

    train(model, args.epochs, dataloader, criterion)

    print("Done training")

    """# Save Model to S3"""


    model_path = os.path.join(args.model_dir, 'model.pth')

    torch.save(model.state_dict(), model_path)

    # torch.save(model.state_dict(), 'model.pth')

    #sage maker automatically uploads artifacts from SM_MODEL_DIR to the output path

    # s3 = boto3.client(region_name='us-east-1', service_name='s3')
    # s3.upload_file(model_path, args.s3_bucket, args.s3_output_key)
    # print(f"Model saved to s3://{args.s3_bucket}/{args.s3_output_key}")
