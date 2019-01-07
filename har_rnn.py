"""
Tested with data_preprocess.py from HAR_CNN repository.
There are still some issue with size. Detailed explanation see below.
"""

import torch
import torch.nn as nn



class HAR_RNN(nn.Module):



    def __init__(self, input_size=9, output_size=6, hidden_size=9, layers_count=2):

        super(HAR_RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.RNN(input_size, hidden_size, layers_count, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)



    def forward(self, x, hidden_data):

        raw_output, hidden_data = self.rnn(x, hidden_data)
        output = self.fc(raw_output)
        
        return output, hidden_data

"""
When tried to train the model, I got error message:
The size of tensor a (6) must match the size of tensor b (64) at non-singleton dimension 2

Unfortunately I couldn't figure out how to solve this without creating even greater errors.
"""
