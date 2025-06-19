import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, vocab_size, num_heads=8, num_layers=4, dropout=0.1, max_len=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))  # positional embeddings

    def forward(self, tgt, memory):
        # Embed target tokens and add positional embeddings
        tgt = self.embedding(tgt) + self.pos_embed[:, :tgt.size(1)]
        tgt = tgt.transpose(0, 1)  # Transformer expects (sequence_length, batch_size, embed_dim)
        memory = memory.unsqueeze(0)  # (1, batch_size, embed_dim)
        output = self.transformer_decoder(tgt, memory)
        output = output.transpose(0, 1)  # back to (batch_size, sequence_length, embed_dim)
        return self.fc_out(output)  # project to vocabulary size logits
