import torch
import torch.nn as nn
import torch.nn.functional as F

# Seq2Seq Model Components: Encoder, Decoder, Seq2Seq
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.3):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        outputs, (hidden, cell) = self.lstm(x, (h0, c0))
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, dropout=0.3):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc_out(output)
        return prediction, hidden, cell
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout=0.3):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(output_dim, hidden_dim, num_layers, dropout)

    def forward(self, src, target_len):
        batch_size = src.size(0)
        input_dim = src.size(2)
        hidden, cell = self.encoder(src)
        decoder_input = torch.zeros(batch_size, 1, input_dim).to(src.device)
        predictions = torch.zeros(batch_size, target_len, input_dim).to(src.device)
        for t in range(target_len):
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell)
            predictions[:, t:t+1, :] = prediction
            decoder_input = prediction
        return predictions