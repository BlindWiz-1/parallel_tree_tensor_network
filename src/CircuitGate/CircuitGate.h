#ifndef CIRCUITGATE_H
#define CIRCUITGATE_H

#include <Eigen/Dense>
#include <vector>

class CircuitGate {
public:
    using Matrix = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>;

    CircuitGate(const Matrix& gate_matrix, const std::vector<int>& sites);

    const Matrix& getGateMatrix() const;
    const std::vector<int>& getSites() const;
    int getDimension() const;

    void display() const;

private:
    Matrix gate_matrix_;
    std::vector<int> sites_;
    int dim_;
};

#endif //CIRCUITGATE_H
