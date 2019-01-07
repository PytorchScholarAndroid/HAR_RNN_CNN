"""
Tested with data_preprocess.py from HAR_CNN repository.

DataLoader here produces data with batch size of 64. This model is batch size agnostic now.

Input size 1152 of final classifier is 128 * 9 (timesteps * inputs_per_timestep)

"""

import torch
import torch.nn as nn



class HAR_RNN(nn.Module):



    def __init__(self, input_size=9, output_size=6, hidden_size=9, layers_count=2):

        super(HAR_RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.RNN(input_size, hidden_size, layers_count, batch_first=True)

        self.fc = nn.Linear(1152, output_size)



    def forward(self, x, hidden_data):

        raw_output, hidden_data = self.rnn(x, hidden_data)
        raw_output = raw_output.reshape(raw_output.shape[0], raw_output.shape[1] * raw_output.shape[2])
        output = self.fc(raw_output)
        
        return output, hidden_data


    
"""

Even though I'm not sure this is the top most good solution of this problem, it works just fine.

Here is a code how to train:

"""

def train(model, dataloader, epoch_count, criterion, optimizer):

    global batch_size, device, hidden_size, hidden_layers, inputs_per_timestep

    for epoch in range(epoch_count):

        model.train()

        # Preparing hidden_content for the first batch to avoid errors
        hidden_content = torch.zeros(hidden_layers, batch_size, inputs_per_timestep)
        hidden_content = hidden_content.to(device)

        for data, labels in train_loader:

            if data.shape[0] != batch_size:
                hidden_content = torch.zeros(hidden_layers, data.shape[0], inputs_per_timestep)
                hidden_content = hidden_content.to(device)

            data = data.to('cuda')
            labels = torch.tensor(labels)
            labels = labels.to(device)
            data = data.float()

            # Loader loads shape batch_szie * timesteps * inputs_per_timestep * 1
            data = torch.squeeze(data, 3)
            # New shape is batch_szie * timesteps * inputs_per_timestep

            prediction, hidden_content = model(data, hidden_content)
            loss = criterion(prediction, labels)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            print('{}/{} Loss: '.format(epoch+1, epoch_count), loss.item())

    return model



batch_size = 64
device = 'cpu'
hidden_layers = 2
hidden_size = 9
inputs_per_timestep = 9
learning_rate = 0.02
timesteps = 128
train_loader, valid_loader, test_loader = load(batch_size)



if __name__ == '__main__':

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = HAR_RNN()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    model = train(model, train_loader, 10, criterion, optimizer)


