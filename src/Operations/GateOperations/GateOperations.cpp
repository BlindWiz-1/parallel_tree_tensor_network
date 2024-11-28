#include "GateOperations.h"
#include <cassert>
#include <iostream>
#include "../../TTNCircuitSim/TTN/TTN.h"
#include <chrono>
#include "GateOperations.h"
#include <cassert>
#include <iostream>
#include "../../TTNCircuitSim/TTN/TTN.h"
#include <chrono>
#include <Eigen/Core>
#include <cmath>

void GateOperations::applySingleParticleGate(TTN& psi, const Eigen::MatrixXcd& gate_matrix, int site) {
    assert(0 <= site && site < psi.nSites());
    psi.getTNodeRoot()->getItem(std::to_string(site))->applyGate(gate_matrix);
}

// Function to reshape a 2D matrix to a 4D-like structure and transpose it accordingly
Eigen::MatrixXcd reshapeAndTransposeMatrix(const Eigen::MatrixXcd& matrix, int local_dimension) {
    // Reshape matrix to 4D-like structure by manually rearranging the elements
    Eigen::MatrixXcd reshaped_matrix(local_dimension * local_dimension, local_dimension * local_dimension);
    for (int i = 0; i < local_dimension; ++i) {
        for (int j = 0; j < local_dimension; ++j) {
            for (int k = 0; k < local_dimension; ++k) {
                for (int l = 0; l < local_dimension; ++l) {
                    reshaped_matrix(i * local_dimension + k, j * local_dimension + l) =
                        matrix(i * local_dimension + j, k * local_dimension + l);
                }
            }
        }
    }

    return reshaped_matrix;
}

// Function to reshape matrix similar to numpy.reshape (3D reshape)
std::vector<Eigen::MatrixXcd> reshapeMatrixTo3D(const Eigen::MatrixXcd& matrix, int dim1, int dim2, int dim3) {
    assert(matrix.size() == dim1 * dim2 * dim3);
    std::vector<Eigen::MatrixXcd> tensor3D(dim3, Eigen::MatrixXcd(dim1, dim2));
    int index = 0;

    for (int k = 0; k < dim3; ++k) {
        for (int i = 0; i < dim1; ++i) {
            for (int j = 0; j < dim2; ++j) {
                tensor3D[k](i, j) = matrix(index);
                ++index;
            }
        }
    }
    return tensor3D;
}

std::tuple<std::vector<Eigen::MatrixXcd>, Eigen::VectorXd, std::vector<Eigen::MatrixXcd>> GateOperations::decomposeTwoParticleGate(const Eigen::MatrixXcd& gate_matrix, int local_dimension) {
    // Step 1: Reshape and transpose the gate_matrix
    Eigen::MatrixXcd reshaped_gate_matrix = reshapeAndTransposeMatrix(gate_matrix, local_dimension);

    // Step 2: Perform SVD on the reshaped matrix
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(reshaped_gate_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXcd u_matrix = svd.matrixU();
    Eigen::MatrixXcd v_matrix = svd.matrixV().adjoint();  // Equivalent to transpose in complex space
    Eigen::VectorXd singular_values = svd.singularValues();

    // Step 3: Filter small singular values
    Eigen::VectorXd::Index num_nonzero = (singular_values.array() > 1e-14).count();
    u_matrix = u_matrix.leftCols(num_nonzero);
    v_matrix = v_matrix.leftCols(num_nonzero);
    singular_values = singular_values.head(num_nonzero);

    std::vector<Eigen::MatrixXcd> u_tensor = reshapeMatrixTo3D(u_matrix, local_dimension, local_dimension, num_nonzero);
    std::vector<Eigen::MatrixXcd> v_tensor = reshapeMatrixTo3D(v_matrix, local_dimension, local_dimension, num_nonzero);

    return std::make_tuple(u_tensor, singular_values, v_tensor);
}

void printTensorDimensions(const std::variant<Eigen::MatrixXcd, Eigen::Tensor<std::complex<double>, 3>>& tensor) {
    if (std::holds_alternative<Eigen::MatrixXcd>(tensor)) {
        const auto& matrix = std::get<Eigen::MatrixXcd>(tensor);
        std::cout << "(" << matrix.rows() << "x" << matrix.cols() << ")" << std::endl;
    } else if (std::holds_alternative<Eigen::Tensor<std::complex<double>, 3>>(tensor)) {
        const auto& tensor_3d = std::get<Eigen::Tensor<std::complex<double>, 3>>(tensor);
        std::cout << "(" << tensor_3d.dimension(0) << "x" << tensor_3d.dimension(1) << "x" << tensor_3d.dimension(2) << ")" << std::endl;
    } else {
        std::cerr << "Unknown tensor type!" << std::endl;
    }
}

void GateOperations::applyTwoParticleGate(TTN& psi, const Eigen::MatrixXcd& gate_matrix, int site_i, int site_j) {
    assert(site_i >= 0 && site_i < site_j && site_j < psi.nSites());

    // Decompose the two-particle gate into U, singular values, and V
    std::vector<Eigen::MatrixXcd> u_matrix, v_matrix;
    Eigen::VectorXd singular_values;
    std::tie(u_matrix, singular_values, v_matrix) = decomposeTwoParticleGate(gate_matrix, psi.localDim());

    // Scale U and V by sqrt(singular_values)
    for (int i = 0; i < singular_values.size(); ++i) {
        std::complex<double> scale_factor = std::complex<double>(std::sqrt(singular_values[i]), 0.0);
        u_matrix[i] *= scale_factor;
        v_matrix[i] *= scale_factor;
    }

    auto node_i = psi.getTNodeRoot()->getItem(std::to_string(site_i));
    auto node_j = psi.getTNodeRoot()->getItem(std::to_string(site_j));

    // Logging tensor dimensions before applying gate
    std::cout << "Tensor at site " << site_i << " dimensions: ";
    printTensorDimensions(node_i->getTensor());
    std::cout << "Tensor at site " << site_j << " dimensions: ";
    printTensorDimensions(node_j->getTensor());

    // Apply gate and reshape
    node_i->applyGateAndReshape(u_matrix);
    node_j->applyGateAndReshape(v_matrix);

    // Logging tensor dimensions after applying gate
    std::cout << "Tensor at site " << site_i << " after reshaping: ";
    printTensorDimensions(node_i->getTensor());
    std::cout << "Tensor at site " << site_j << " after reshaping: ";
    printTensorDimensions(node_j->getTensor());

    // Dimension check
    int gate_dim = singular_values.size();
    assert(gate_dim > 0 && "Gate dimension after SVD truncation should be non-zero");

    // Wire all common nodes on the path from i to j (excluding the leaves)
    auto nodes = psi.getTNodeRoot()->getIntermediateNodes(std::to_string(site_i), std::to_string(site_j));
    for (auto& node : nodes) {

        std::cout << "Updating intermediate node " << node->getName() << " with gate_dim " << gate_dim << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        node->update(gate_dim, site_i, site_j);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        if (duration > 10000) {
            std::cerr << "Wiring of " << node->getName() << " with shape ";
            printTensorDimensions(node->getTensor());
            std::cerr << " took " << duration / 1000 << " seconds" << std::endl;
        } else {
            std::cout << "Wiring of " << node->getName() << " with shape ";
            printTensorDimensions(node->getTensor());
            std::cout << " took " << duration / 1000 << " seconds" << std::endl;
        }
    }
}

void GateOperations::applyCircuit(TTN& psi, const Circuit& circ) {
    assert(psi.nSites() == circ.getLSites());

    for (size_t idx = 0; idx < circ.getGates().size(); ++idx) {
        const auto& gate = circ.getGates()[idx];
        std::stringstream ss;
        ss << "Gate " << idx << " on ";
        for (int site : gate.getSites()) {
            ss << site << " ";
        }
        std::cout << ss.str() << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        assert(gate.getSites().size() == 1 || gate.getSites().size() == 2);
        if (gate.getSites().size() == 1) {
            applySingleParticleGate(psi, gate.getGateMatrix(), gate.getSites()[0]);
        } else {
            applyTwoParticleGate(psi, gate.getGateMatrix(), gate.getSites()[0], gate.getSites()[1]);
            psi.orthonormalize(gate.getSites()[0], gate.getSites()[1]);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        std::cout << "Gate " << idx << " on ";
        for (int site : gate.getSites()) {
            std::cout << site << " ";
        }
        std::cout << "took " << duration.count() << " seconds" << std::endl;
    }
}
