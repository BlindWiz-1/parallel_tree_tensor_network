#include <iostream>
#include <vector>
#include <optional>
#include <Eigen/Dense>
#include "TTNCircuitSim/TTN/TTN.h"
#include "TTNCircuitSim/TNode/TNode.h"
#include "Structure/SNode/SNode.h"
#include "Circuits/Circuit/Circuit.h"
#include "Circuits/QuantumGate/QuantumGate.h"
#include "Circuits/SingleStateToTrees/SingleStateToTrees.h"
#include "Operations/GateOperations/GateOperations.h"

int main() {
    const int d = 2;

    // Start from computational basis state
    const std::vector<int> single_states = {0, 0, 0};

    Circuit circ(4, d);

    circ.appendGate(CircuitGate(QuantumGate::H, {0}));
    circ.appendGate(CircuitGate(QuantumGate::CNOT, {0, 1}));
    circ.appendGate(CircuitGate(QuantumGate::CNOT, {0, 2}));

    std::shared_ptr<TTN> psi = TTN::basisState(d, single_states, nullptr, circ, 100, true);
    GateOperations::applyCircuit(*psi, circ);

    std::cout << "Number of qubits: " << psi->nSites() << std::endl;
    std::cout << "Output state as vector (should be the GHZ state):" << std::endl;
    std::cout << psi->asVector().transpose() << std::endl;

    return 0;
}
