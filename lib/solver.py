import torch
import torch.optim as optim

class Solver(object):

    def __init__(self, model, learning_rate, weight_decay):
        self.model = model
        self.criterion = torch.nn.BCELoss()  # or any other loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
    def train(self, dataloader):
        self.model.train()  # set the model to training mode
        total_loss = 0
        for batch_idx, (data, targets) in enumerate(dataloader):
            # forward pass
            output = self.model(data)
            loss = self.criterion(output, targets)

            # backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        
        average_loss = total_loss / len(dataloader)
        print('Average Loss: ', average_loss)
        
    def test(self, dataloader):
        self.model.eval()  # set the model to eval mode
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(dataloader):
                output = self.model(data)
                loss = self.criterion(output, targets)
                total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print('Test Loss: ', average_loss)


