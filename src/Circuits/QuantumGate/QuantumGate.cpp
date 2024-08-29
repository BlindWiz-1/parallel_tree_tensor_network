// QuantumGate.cpp
#include "QuantumGate.h"

const Eigen::Matrix2cd QuantumGate::PauliX = (Eigen::Matrix2d() << 0, 1, 1, 0).finished();
const Eigen::Matrix2cd QuantumGate::PauliY = (Eigen::Matrix2cd() << 0, std::complex<double>(0, -1), std::complex<double>(0, 1), 0).finished();
const Eigen::Matrix2cd QuantumGate::PauliZ = (Eigen::Matrix2d() << 1, 0, 0, -1).finished();
const Eigen::Matrix2cd QuantumGate::H = (Eigen::Matrix2d() << 1 / sqrt(2), 1 / sqrt(2), 1 / sqrt(2), -1 / sqrt(2)).finished();

const Eigen::Matrix4cd QuantumGate::ISWAP = (Eigen::Matrix4cd() <<
    1.0, 0, 0, 0,
    0, 0, std::complex<double>(0, 1), 0,
    0, std::complex<double>(0, 1), 0, 0,
    0, 0, 0, 1).finished();

const Eigen::Matrix4cd QuantumGate::ROOT_ISWAP = QuantumGate::computeMatrixSquareRoot(QuantumGate::ISWAP);
const Eigen::Matrix4cd QuantumGate::CNOT = (Eigen::Matrix4d() <<
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 1,
    0, 0, 1, 0).finished();

const Eigen::Matrix2cd QuantumGate::ROOT_X = QuantumGate::computeMatrixSquareRoot((Eigen::Matrix2cd() << 0, 1, 1, 0).finished());
const Eigen::Matrix2cd QuantumGate::ROOT_Y = QuantumGate::computeMatrixSquareRoot((Eigen::Matrix2cd() << 0, std::complex<double>(0, -1), std::complex<double>(0, 1), 0).finished());
const Eigen::Matrix2cd QuantumGate::ROOT_Z = QuantumGate::computeMatrixSquareRoot((Eigen::Matrix2cd() << 1, 0, 0, -1).finished());