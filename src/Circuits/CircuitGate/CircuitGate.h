#ifndef CIRCUITGATE_H
#define CIRCUITGATE_H

#include <Eigen/Dense>
#include <vector>

class CircuitGate {
public:
    CircuitGate(const Eigen::MatrixXcd& gate_matrix, const std::vector<int>& sites);

    const Eigen::MatrixXcd& getGateMatrix() const;
    const std::vector<int>& getSites() const;
    int getDimension() const;

    void display() const;

private:
    Eigen::MatrixXcd gate_matrix_;
    std::vector<int> sites_;
    int dim_;
};

#endif //CIRCUITGATE_H
