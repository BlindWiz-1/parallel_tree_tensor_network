#include <iostream>
#include <vector>
#include "Circuits/Circuit/Circuit.h"
#include "Circuits/CircuitGate/CircuitGate.h"
#include "Circuits/QuantumGate/QuantumGate.h"
#include "TTNCircuitSim/TNode/TNode.h"
#include "Circuits/SingleStateToTrees/SingleStateToTrees.h"
#include "Structure/FindTreeStructure/FindTreeStructure.h"

int main() {
    int l_sites = 4;
    int local_dimension = 2;
    Circuit circuit(l_sites, local_dimension);

    Eigen::Matrix2cd pauliX = QuantumGate::PauliX;
    Eigen::Matrix2cd pauliY = QuantumGate::PauliY;

    circuit.appendGate(CircuitGate(pauliX, {0}));
    circuit.appendGate(CircuitGate(pauliY, {1}));
    circuit.appendGate(CircuitGate(pauliX, {2}));
    circuit.appendGate(CircuitGate(pauliY, {3}));
    circuit.appendGate(CircuitGate(pauliX, {0, 1}));
    circuit.appendGate(CircuitGate(pauliY, {2, 3}));

    std::cout << "Circuit:" << std::endl;
    circuit.display();

    auto treeStructure = findTreeStructure(circuit);

    std::vector<int> single_states = {0, 0, 0, 0};
    auto root_node = singleStatesToTree(single_states, local_dimension, treeStructure);

    std::cout << "\nTree Structure:" << std::endl;
    root_node->display();

    return 0;
}
