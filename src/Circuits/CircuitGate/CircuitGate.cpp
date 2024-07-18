#include "CircuitGate.h"
#include <iostream>

CircuitGate::CircuitGate(const Matrix& gate_matrix, const std::vector<int>& sites)
: gate_matrix_(gate_matrix), sites_(sites), dim_(gate_matrix.rows()) {
    assert(!sites.empty());
    assert(gate_matrix.rows() == gate_matrix.cols());
}

const CircuitGate::Matrix& CircuitGate::getGateMatrix() const {
    return gate_matrix_;
}

const std::vector<int>& CircuitGate::getSites() const {
    return sites_;
}

int CircuitGate::getDimension() const {
    return dim_;
}

void CircuitGate::display() const {
    std::cout << "Gate Matrix: \n" << gate_matrix_ << "\n";
    std::cout << "Sites: ";
    for (const int site : sites_) {
        std::cout << site << " ";
    }
    std::cout << "\n";
}
