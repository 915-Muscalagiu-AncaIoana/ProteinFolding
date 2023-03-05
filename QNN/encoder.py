import torch
from torch import Tensor
import qiskit as qk


def compute_masks(state_length, action_no):
    no_qubits_per_action = state_length // action_no
    masks = []
    formatt = "0" + str(state_length) + "b"
    for mask_no in range(action_no):
        mask = []
        for number in range(pow(2, state_length)):
            binary_form = format(number, formatt)
            binary_form = binary_form[::-1]
            start = no_qubits_per_action * mask_no
            end = no_qubits_per_action * (mask_no + 1)
            sum = 0
            for index in range(start, end):
                sum += int(binary_form[index])
            mask.append(pow(-1,sum))
        masks.append(mask)
    return masks

def encoding_circuit(inputs, num_qubits, *args):
    """
    Encode classical input data (i.e. the state of the enironment) on a quantum circuit.
    To be used inside the `parametrized_circuit` function.

    Args
    -------
    inputs (list): a list containing the classical inputs.
    num_qubits (int): number of qubits in the quantum circuit.

    Return
    -------
    qc (QuantumCircuit): quantum circuit with encoding gates.

    """

    qc = qk.QuantumCircuit(num_qubits)

    # Encode data with a RX rotation
    for i in range(len(inputs)):
        qc.rx(inputs[i], i)

    return qc


class encoding_layer(torch.nn.Module):
    def __init__(self, num_qubits):
        super().__init__()

        # Define weights for the layer
        weights = torch.Tensor(num_qubits)
        self.weights = torch.nn.Parameter(weights)
        torch.nn.init.uniform_(self.weights, -1, 1)  # <--  Initialization strategy

    def forward(self, x):
        """Forward step, as explained above."""

        x = self.weights * x
        x = torch.atan(x)

        return x


class exp_val_layer(torch.nn.Module):
    def __init__(self, action_space, state_length):
        super().__init__()

        # Define the weights for the layer
        weights = torch.Tensor(action_space)
        self.weights = torch.nn.Parameter(weights)
        torch.nn.init.uniform_(self.weights, 35, 40)  # <-- Initialization strategy (heuristic choice)

        # Masks that map the vector of probabilities to <Z_0*Z_1> and <Z_2*Z_3>

        self.masks = [ torch.tensor(mask, requires_grad=False) for mask in compute_masks(state_length,action_space)]


    def forward(self, x):
        """Forward step, as described above."""

        expvalues = []
        for mask in self.masks:
            expval = mask * x
            expvalues.append(expval)

        # Single sample
        if len(x.shape) == 1:
            final = []
            for expval in expvalues:
                expval = torch.sum(expval)
                expval = expval.unsqueeze(0)
                final.append(expval)
            out = torch.cat(final)
        # Batch of samples
        else:
            final = []
            for expval in expvalues:
                expval = torch.sum(expval, dim=1, keepdim=True)
                final.append(expval)
            out = torch.cat(final, 1)
        return out

