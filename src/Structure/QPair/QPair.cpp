#include "QPair.h"

QPair::QPair(std::pair<int, int> qubits, double similarity)
    : qubits_(qubits), similarity_(similarity) {}

std::pair<int, int> QPair::getQubits() const {
    return qubits_;
}

double QPair::getSimilarity() const {
    return similarity_;
}

bool QPair::operator<(const QPair& other) const {
    if (similarity_ == other.similarity_) {
        return qubits_ < other.qubits_;
    }
    return similarity_ > other.similarity_;
}
