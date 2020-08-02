import torch
import config as c
from torch import nn
import numpy as np
import crop_dataset as data
from torch.nn import Module, Conv2d


def evaluate_model(model, data_loader, device):
    mses = np.zeros(shape=len(data_loader))
    sample_num = 0

    with torch.no_grad():
        for X, y, meta in data_loader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            y = y.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            mses[sample_num] = np.mean((y[0] - outputs[0]) ** 2)
            sample_num += 1
    return np.mean(mses)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    learning_rate = 1e-3
    weight_decay = 1e-5

    run_validation_batch_num = 1
    print_patch_num = 100
    n_epoch = 4

    model = ConvNet().to(device)
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    test_loader, val_loader, train_loader = data.create_data_loader()

    batch_count = 0
    best_val_loss = np.inf
    sample_count = 0

    for epoch in range(n_epoch):
        for X, y, meta in train_loader:
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            optimizer.zero_grad()
            loss = mse(outputs, y)
            loss.backward()
            optimizer.step()
            sample_count += X.shape[0]

            if (batch_count + 1) % print_patch_num == 0:
                print(f'Epoch[{epoch + 1}/{n_epoch}] Batch [{batch_count + 1}], Loss: {loss.item():.4f}')
            if (batch_count + 1) % run_validation_batch_num == 0:
                val_loss = evaluate_model(model, test_loader, device)
                val_loss_nom = val_loss.item()
                print(f'Validation set, Loss: {val_loss_nom:.4f}')
                if val_loss_nom <= best_val_loss:
                    best_val_loss = val_loss_nom
                    torch.save(model, c.BEST_MODEL_FILE)
            batch_count += 1

    test_loss = evaluate_model(model, test_loader, device)
    val_loss = evaluate_model(model, val_loader, device)
    train_loss = evaluate_model(model, train_loader, device)

    with open(c.BEST_RESULTS_FILE, 'w') as fh:
        print(f"Scores:", file=fh)
        print(f"Test Loss: {test_loss.item():.4f}", file=fh)
        print(f"Val Loss: {val_loss.item():.4f}", file=fh)
        print(f"Train Loss: {train_loss.item():.4f}", file=fh)

    print('Finished Training')


class ConvNet(nn.Module):
    def __init__(self, n_in_channels=1, n_hidden_layers=3, n_kernels=32, kernel_size=7):
        super(ConvNet, self).__init__()

        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(in_channels=n_in_channels, out_channels=n_kernels, kernel_size=kernel_size,
                                       bias=True, padding=int(kernel_size / 2)))
            cnn.append(torch.nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)
        self.output_layer = torch.nn.Conv2d(in_channels=n_in_channels, out_channels=1,
                                            kernel_size=kernel_size, bias=True, padding=int(kernel_size / 2))

    def forward(self, x):
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        cnn_out = self.hidden_layers(x)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        pred = self.output_layer(cnn_out)  # apply output layer (N, n_kernels, X, Y) -> (N, 1, X, Y)
        return pred
