import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from collections import Counter

"""# dataset preparation"""

# Dataset Preparation
with open('datasets/alice_1.txt', 'r', encoding='utf-8') as file:
    text = file.read()
# tokenize the text into words
words = text.split()
# count unique words from text
word_counts = Counter(words)
# make list of the unique words ---> to create a vocabulary
vocab = list(word_counts.keys())
vocab_size = len(vocab)
word_to_int = {word: i for i, word in enumerate(vocab)} # maps each word to a unique integer index
int_to_word = {i: word for word, i in word_to_int.items()} # maps each integer to a word
SEQUENCE_LENGTH = 64
samples = [words[i:i+SEQUENCE_LENGTH+1] for i in range(len(words)-SEQUENCE_LENGTH)] # training samples of 64 word length
print(vocab)
print(word_to_int)
print(int_to_word)

"""# creating dataset class and data loader"""

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

BATCH_SIZE = 32
dataset = TextDataset(samples, word_to_int) # making dataset
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
) # creating batches from dataset
input, output = dataset[1]
print("input length: ", len(input))
print("output length: ", len(output))
print(dataset[1]) # one sample from dataset --> 2nd sample. returns input and then returns output

"""# causal masking and positional encoding methods
causal masking --> generates triangular mask. makes -inf for masked positions --> softmax(-inf) = 0

positional encoding --> not sure why dropout is needed?
"""

#CAUsAL MASKING
def generate_square_subsequent_mask(sz):
  """
  Generate a square mask for the sequence. The masked positions are filled with float('-inf').
  Unmasked positions are filled with float(0.0).
  """
  mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
  mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
  return mask

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

"""# decoder-only transformer model
- `vocab_size` = number of unique words in vocabulary
- `embed_dim` = size of embedding vector
- `num_layers` = number of instances of specified decoders (stacked together)

"""

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

"""# training - hyperparameters & initialize model"""

epochs = 50 # temp epochsrhgug
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(
    vocab_size=vocab_size,
    embed_dim=100,
    num_layers=2,
    num_heads=2,
).to(device) # have to shift model to the device
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(model)
# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.\n")

"""# training method"""

# Training
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
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch} loss: {epoch_loss:.3f}")

train(model, epochs, dataloader, criterion)

""""One thing to note in the training loop is the shape of the targets and the outputs before calculating the loss. We need to ensure that the shape of the targets is [batch_size x sequence_length] in flattened format and the shape of the outputs is [batch_size x sequence_length, vocab_size]."

# inference
"""

def return_int_vector(text):
    words = text.split()
    input_seq = torch.LongTensor([word_to_int[word] for word in words[-SEQUENCE_LENGTH:]]).unsqueeze(0)
    return input_seq

def sample_next(predictions, temperature=1.0, top_k=None):
    """
    Sample the next token using temperature and top-k sampling.

    :param predictions: Model logits for the next word.
    :param temperature: Controls randomness (higher = more random).
    :param top_k: If set, restricts sampling to top-k most likely words.
    """
    probabilities = F.softmax(predictions[:, -1, :] / temperature, dim=-1).cpu()

    if top_k is not None:
        # Select top-k probabilities
        top_values, top_indices = torch.topk(probabilities, top_k)
        probabilities = top_values / torch.sum(top_values)  # Re-normalize
        next_token = torch.multinomial(probabilities, 1).item()
        next_token = top_indices[next_token].item()  # Convert to actual token index
    else:
        # Sample from full distribution
        next_token = torch.multinomial(probabilities, 1).item()

    return next_token

def text_generator(sentence, generate_length, temperature=1.0, top_k=None):
    model.eval()
    sample = sentence
    for i in range(generate_length):
        int_vector = return_int_vector(sample)
        if len(int_vector) >= SEQUENCE_LENGTH - 1:
            break
        input_tensor = int_vector.to(device)
        with torch.no_grad():
            predictions = model(input_tensor)
        next_token = sample_next(predictions, temperature, top_k)
        sample += ' ' + int_to_word[next_token]
    print(sample)
    print('\n')

"""# sample inference"""

sentences = ["Alice was sleepy"]
generate_length = 100
temperature = 1.0
for sentence in sentences:
    print(f"PROMPT: {sentence}")
    temperature = 1.0
    print("TEMPERATURE " + str(temperature) + ":")
    text_generator(sentence, generate_length, temperature)
    temperature = 0.5
    print("TEMPERATURE " + str(temperature) + ":")
    text_generator(sentence, generate_length, temperature)
    temperature = 0.2
    print("TEMPERATURE " + str(temperature) + ":")
    text_generator(sentence, generate_length, temperature)