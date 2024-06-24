#include <iostream>
#include <Eigen/Dense>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

using namespace Eigen;

// Define a simple quantum gate (e.g., Pauli-X) using Eigen
Matrix2d pauliX() {
    Matrix2d X;
    X << 0, 1,
         1, 0;
    return X;
}

// Function to create a basic tensor network from a quantum gate using Xtensor
xt::xarray<double> createTensorFromGate(const Matrix2d& gate) {
    xt::xarray<double> tensor = {{gate(0,0), gate(0,1)},
                                 {gate(1,0), gate(1,1)}};
    return tensor;
}

int main() {
    // Example using Eigen for matrix operations
    Matrix2d X = pauliX();
    std::cout << "Pauli-X gate (Eigen):\n" << X << std::endl;

    // Example using Xtensor for tensor network operations
    auto TX = createTensorFromGate(X);
    std::cout << "Tensor from Pauli-X gate (Xtensor):\n" << TX << std::endl;

    // Further tensor network operations can be performed here

    return 0;
}
