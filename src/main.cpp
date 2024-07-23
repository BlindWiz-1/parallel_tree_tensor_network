#include <iostream>
#include <vector>
#include <memory>
#include "Circuits/Circuit/Circuit.h"
#include "Circuits/CircuitGate/CircuitGate.h"
#include "Circuits/QuantumGate/QuantumGate.h"
#include "TTNCircuitSim/TNode/TNode.h"
#include "Structure/SNode/SNode.h"
#include "Circuits/SingleStateToTrees/SingleStateToTrees.h"
#include "Structure/FindTreeStructure/FindTreeStructure.h"

int main() {
    // Step 1: Create a Circuit object
    int l_sites = 4;  // Number of qubits/sites
    int local_dimension = 2;  // Dimension of each qubit (2 for a standard qubit)
    Circuit circuit(l_sites, local_dimension);

    // Step 2: Add gates to the circuit
    Eigen::Matrix2cd pauliX = QuantumGate::PauliX;
    Eigen::Matrix2cd pauliY = QuantumGate::PauliY;

    // Create some CircuitGate objects and add them to the circuit
    circuit.appendGate(CircuitGate(pauliX, {0}));
    circuit.appendGate(CircuitGate(pauliY, {1}));
    circuit.appendGate(CircuitGate(pauliX, {2}));
    circuit.appendGate(CircuitGate(pauliY, {3}));
    circuit.appendGate(CircuitGate(pauliX, {0, 1}));
    circuit.appendGate(CircuitGate(pauliY, {2, 3}));

    // Display the circuit
    std::cout << "Circuit:" << std::endl;
    circuit.display();

    // Step 3: Convert the circuit to a tree structure
    auto treeStructure = findTreeStructure(circuit);

    // Step 4: Convert the tree structure to a TNode
    std::vector<int> single_states = {0, 0, 0, 0};  // Basis state for each qubit
    auto root_node = singleStatesToTree(single_states, local_dimension, treeStructure);

    // Step 5: Display the tree structure
    std::cout << "\nTree Structure:" << std::endl;
    root_node->display();

    return 0;
}
