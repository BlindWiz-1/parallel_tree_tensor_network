#include <iostream>
#include <vector>
#include <optional>
#include "TTNCircuitSim/TTN/TTN.h"
#include "Circuits/Circuit/Circuit.h"
#include "Circuits/QuantumGate/QuantumGate.h"
#include "Circuits/SingleStateToTrees/SingleStateToTrees.h"
#include "Operations/GateOperations/GateOperations.h"
#include <Eigen/Core>

int main() {
    Eigen::initParallel();
    const int d = 2;

    // Start from computational basis state
    const std::vector<int> single_states = {0, 0, 0};

    Circuit circ(3, d);

    circ.appendGate(CircuitGate(QuantumGate::H, {0}));
    circ.appendGate(CircuitGate(QuantumGate::CNOT, {0, 1}));
    circ.appendGate(CircuitGate(QuantumGate::CNOT, {0, 2}));

    std::shared_ptr<TTN> psi = TTN::basisState(2, single_states, nullptr, circ, 100, false);
    GateOperations::applyCircuit(*psi, circ);

    std::cout << "Number of qubits: " << psi->nSites() << std::endl;
    std::cout << "State of tree after circuit application" << std::endl;
    psi->display();
    std::cout << "Output state as vector (should be the GHZ state):" << std::endl;
    std::cout << psi->asVector().transpose() << std::endl;

    return 0;
}
