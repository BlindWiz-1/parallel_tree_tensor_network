#ifndef QPAIR_H
#define QPAIR_H

#include <utility>

class QPair {
public:
    QPair(std::pair<int, int> qubits, double similarity);

    std::pair<int, int> getQubits() const;
    double getSimilarity() const;

    bool operator<(const QPair& other) const;

private:
    std::pair<int, int> qubits_;
    double similarity_;
};

#endif // QPAIR_H
