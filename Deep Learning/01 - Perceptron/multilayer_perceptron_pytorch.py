import torch
from torch.nn import Sequential

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()

        self.layers = Sequential(

            # Hidden Layer - 01
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # Hidden Layer - 02
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # Output layer
            torch.nn.Linear(20, num_outputs)
        )

    def forward(self, x) -> torch.Tensor:
        logits: torch.Tensor = self.layers(x)
        return logits


if __name__ == "__main__":
    torch.manual_seed(123)
    model: torch.nn.Module = NeuralNetwork(50, 3)

    trainable_parameters: int = sum([parameter.numel() for parameter in model.parameters() if parameter.requires_grad])
    print(f"Number of trainable parameters: {trainable_parameters}")

    X: torch.Tensor = torch.rand(size=(1, 50))

    with torch.no_grad():
        out: torch.Tensor = model(X)
        out = torch.softmax(input=out, dim=1)

    print(f"Model output: {out}")
