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

int main() {
    const int d = 2;

    // Start from computational basis state
    const std::vector<int> single_states = {0, 0, 0};
    std::shared_ptr<TTN> psi = TTN::basisState(d, single_states, nullptr, Circuit(4, d), {{"d_max", 100}, {"enable_gpu", 1}, {"dry_run", 1}});

    std::cout << "Number of qubits: " << psi->nSites() << std::endl;

    // Define some standard gates
    Eigen::Matrix2cd H = (Eigen::Matrix2cd() << 1.0 / std::sqrt(2), 1.0 / std::sqrt(2), 1.0 / std::sqrt(2), -1.0 / std::sqrt(2)).finished();
    Eigen::Matrix4cd Ucnot = (Eigen::Matrix4cd() << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0).finished();

    // Use circuit to prepare the Greenberger-Horne-Zeilinger (GHZ) state
    Circuit circ(4, d);
    circ.appendGate(CircuitGate(H, {0}));
    circ.appendGate(CircuitGate(Ucnot, {0, 1}));
    circ.appendGate(CircuitGate(Ucnot, {0, 2}));

    psi->applyCircuit(circ.getGates());

    std::cout << "Output state as vector (should be the GHZ state):" << std::endl;
    std::cout << psi->asVector().transpose() << std::endl;

    return 0;
}
