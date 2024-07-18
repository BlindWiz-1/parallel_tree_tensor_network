// QuantumGate.h
#ifndef QUANTUMGATE_H
#define QUANTUMGATE_H

#include <Eigen/Dense>
#include <iostream>

class QuantumGate {
public:
    static const Eigen::Matrix2d PauliX;
    static const Eigen::Matrix2cd PauliY;
    static const Eigen::Matrix2d PauliZ;
    static const Eigen::Matrix2d H;

    static const Eigen::Matrix4cd ISWAP;
    static const Eigen::Matrix4cd ROOT_ISWAP;
    static const Eigen::Matrix4d CNOT;

    static const Eigen::Matrix2cd ROOT_X;
    static const Eigen::Matrix2cd ROOT_Y;
    static const Eigen::Matrix2cd ROOT_Z;

    template<typename MatrixType>
    static void display(const MatrixType& gate);

    template<typename MatrixType>
    static MatrixType computeMatrixSquareRoot(const MatrixType& matrix);

private:
    QuantumGate() {}
};

template<typename MatrixType>
void QuantumGate::display(const MatrixType& gate) {
    std::cout << gate << std::endl;
}

// Template function to compute matrix square root
template<typename MatrixType>
MatrixType QuantumGate::computeMatrixSquareRoot(const MatrixType& matrix) {
    Eigen::ComplexSchur<MatrixType> schur(matrix);
    MatrixType U = schur.matrixU();
    MatrixType T = schur.matrixT();

    for (int i = 0; i < T.rows(); ++i) {
        T(i, i) = std::sqrt(T(i, i));
    }

    return U * T * U.adjoint();
}

#endif //QUANTUMGATE_H
