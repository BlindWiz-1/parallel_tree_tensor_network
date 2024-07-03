#include "QuantumGate/QuantumGate.h"
#include "CircuitGate/CircuitGate.h"
#include "Circuit/Circuit.h"

using namespace Eigen;

int main() {
    CircuitGate pauli_x(QuantumGate::ROOT_X, {0});
    CircuitGate cnot(QuantumGate::CNOT, {0, 1});
    CircuitGate root_iswap(QuantumGate::ROOT_ISWAP, {1, 2});

    Circuit circuit(3, 2);  // 3 qubits with a local dimension of 2

    circuit.appendGate(pauli_x);
    circuit.appendGate(cnot);
    circuit.appendGate(root_iswap);

    circuit.display();

    return 0;
}
