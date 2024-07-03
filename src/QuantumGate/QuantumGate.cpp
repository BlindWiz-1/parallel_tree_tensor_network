// QuantumGate.cpp
#include "QuantumGate.h"
#include <iostream>

const Eigen::Matrix2d QuantumGate::PauliX = (Eigen::Matrix2d() << 0, 1, 1, 0).finished();
const Eigen::Matrix2cd QuantumGate::PauliY = (Eigen::Matrix2cd() << 0, std::complex<double>(0, -1), std::complex<double>(0, 1), 0).finished();
const Eigen::Matrix2d QuantumGate::PauliZ = (Eigen::Matrix2d() << 1, 0, 0, -1).finished();
const Eigen::Matrix2d QuantumGate::H = (Eigen::Matrix2d() << 1 / sqrt(2), 1 / sqrt(2), 1 / sqrt(2), -1 / sqrt(2)).finished();

const Eigen::Matrix4cd QuantumGate::ISWAP = (Eigen::Matrix4cd() <<
    1.0, 0, 0, 0,
    0, 0, std::complex<double>(0, 1), 0,
    0, std::complex<double>(0, 1), 0, 0,
    0, 0, 0, 1).finished();

const Eigen::Matrix4cd QuantumGate::ROOT_ISWAP = QuantumGate::computeMatrixSquareRoot(QuantumGate::ISWAP);
const Eigen::Matrix4cd QuantumGate::CNOT = (Eigen::Matrix4cd() <<
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 1,
    0, 0, 1, 0).finished();

const Eigen::Matrix2cd QuantumGate::ROOT_X = QuantumGate::computeMatrixSquareRoot((Eigen::Matrix2cd() << 0, 1, 1, 0).finished());
const Eigen::Matrix2cd QuantumGate::ROOT_Y = QuantumGate::computeMatrixSquareRoot((Eigen::Matrix2cd() << 0, std::complex<double>(0, -1), std::complex<double>(0, 1), 0).finished());
const Eigen::Matrix2cd QuantumGate::ROOT_Z = QuantumGate::computeMatrixSquareRoot((Eigen::Matrix2cd() << 1, 0, 0, -1).finished());

template<typename Derived>
void QuantumGate::display(const Eigen::MatrixBase<Derived>& gate) {
    std::cout << gate << std::endl;
}

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