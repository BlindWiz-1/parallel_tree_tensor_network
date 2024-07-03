// QuantumGate.h
#ifndef QUANTUMGATE_H
#define QUANTUMGATE_H

#include <Eigen/Dense>

class QuantumGate {
public:
    static const Eigen::Matrix2d PauliX;
    static const Eigen::Matrix2cd PauliY;
    static const Eigen::Matrix2d PauliZ;
    static const Eigen::Matrix2d H;

    static const Eigen::Matrix4cd ISWAP;
    static const Eigen::Matrix4cd ROOT_ISWAP;
    static const Eigen::Matrix4cd CNOT;

    static const Eigen::Matrix2cd ROOT_X;
    static const Eigen::Matrix2cd ROOT_Y;
    static const Eigen::Matrix2cd ROOT_Z;

    template<typename Derived>
    static void display(const Eigen::MatrixBase<Derived>& gate);

private:
    QuantumGate() {}

    template<typename MatrixType>
    static MatrixType computeMatrixSquareRoot(const MatrixType& matrix);
};

#endif //QUANTUMGATE_H
