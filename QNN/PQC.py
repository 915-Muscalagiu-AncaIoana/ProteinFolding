import qiskit as qk
from qiskit.circuit.library import TwoLocal

from QNN.encoder import encoding_circuit


def parametrized_circuit(num_qubits=4, reuploading=False, reps=2, insert_barriers=True, meas=False):
    """
    Create the Parameterized Quantum Circuit (PQC) for estimating Q-values.
    It implements the architecure proposed in Skolik et al. arXiv:2104.15084.

    Args
    -------
    num_qubit (int): number of qubits in the quantum circuit.
    reuploading (bool): True if want to use data reuploading technique.
    reps (int): number of repetitions (layers) in the variational circuit.
    insert_barrirerd (bool): True to add barriers in between gates, for better drawing of the circuit.
    meas (bool): True to add final measurements on the qubits.

    Return
    -------
    qc (QuantumCircuit): the full parametrized quantum circuit.
    """

    qr = qk.QuantumRegister(num_qubits, 'qr')
    qc = qk.QuantumCircuit(qr)

    if meas:
        qr = qk.QuantumRegister(num_qubits, 'qr')
        cr = qk.ClassicalRegister(num_qubits, 'cr')
        qc = qk.QuantumCircuit(qr, cr)

    if not reuploading:

        # Define a vector containg Inputs as parameters (*not* to be optimized)
        inputs = qk.circuit.ParameterVector('x', num_qubits)

        # Encode classical input data
        qc.compose(encoding_circuit(inputs, num_qubits=num_qubits), inplace=True)
        if insert_barriers: qc.barrier()

        # Variational circuit
        qc.compose(TwoLocal(num_qubits, ['ry', 'rz'], 'cz', 'circular',
                            reps=reps, insert_barriers=insert_barriers,
                            skip_final_rotation_layer=True), inplace=True)
        if insert_barriers: qc.barrier()

        # Add final measurements
        if meas: qc.measure(qr, cr)
    elif reuploading:

        # Define a vector containg Inputs as parameters (*not* to be optimized)
        inputs = qk.circuit.ParameterVector('x', num_qubits)

        # Define a vector containng variational parameters
        θ = qk.circuit.ParameterVector('θ', 2 * num_qubits * reps)

        # Iterate for a number of repetitions
        for rep in range(reps):

            # Encode classical input data
            qc.compose(encoding_circuit(inputs, num_qubits=num_qubits), inplace=True)
            if insert_barriers: qc.barrier()

            # Variational circuit (does the same as TwoLocal from Qiskit)
            for qubit in range(num_qubits):
                qc.ry(θ[qubit + 2 * num_qubits * (rep)], qubit)
                qc.rz(θ[qubit + 2 * num_qubits * (rep) + num_qubits], qubit)
            if insert_barriers: qc.barrier()

            # Add entanglers (this code is for a circular entangler)
            qc.cz(qr[-1], qr[0])
            for qubit in range(num_qubits - 1):
                qc.cz(qr[qubit], qr[qubit + 1])
            if insert_barriers: qc.barrier()

        # Add final measurements
        if meas: qc.measure(qr, cr)

    return qc