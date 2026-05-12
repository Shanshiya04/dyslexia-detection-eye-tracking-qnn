import pennylane as qml
import torch
import torch.nn as nn

n_qubits = 6
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (3, n_qubits, 3)}
qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

class QNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qnn = qlayer
        self.fc = nn.Linear(n_qubits, 2)

    def forward(self, x):
        x = torch.tanh(x)
        x = self.qnn(x)
        x = self.fc(x)
        return x