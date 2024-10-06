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
    psi.getTNodeRoot()->getItem(site)->applyGate(gate_matrix);
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

    // Step 4: Reshape u_matrix and v_matrix to 3D-like structures using vector of MatrixXcd
    std::cout << "U matrix dimensions before reshape: (" << u_matrix.rows() << "x" << u_matrix.cols() << ")\n";
    std::cout << "V matrix dimensions before reshape: (" << v_matrix.rows() << "x" << v_matrix.cols() << ")\n";

    std::vector<Eigen::MatrixXcd> u_tensor = reshapeMatrixTo3D(u_matrix, local_dimension, local_dimension, num_nonzero);
    std::vector<Eigen::MatrixXcd> v_tensor = reshapeMatrixTo3D(v_matrix, local_dimension, local_dimension, num_nonzero);

    std::cout << "U tensor slices: " << u_tensor.size() << " x " << u_tensor[0].rows() << "x" << u_tensor[0].cols() << "\n";
    std::cout << "V tensor slices: " << v_tensor.size() << " x " << v_tensor[0].rows() << "x" << v_tensor[0].cols() << "\n";

    return std::make_tuple(u_tensor, singular_values, v_tensor);
}

void GateOperations::applyTwoParticleGate(TTN& psi, const Eigen::MatrixXcd& gate_matrix, int site_i, int site_j) {
    assert(site_i >= 0 && site_i < site_j && site_j < psi.nSites());

    // Decompose the two-particle gate into U, singular values, and V
    std::vector<Eigen::MatrixXcd> u_matrix, v_matrix;
    Eigen::VectorXd singular_values;
    std::tie(u_matrix, singular_values, v_matrix) = decomposeTwoParticleGate(gate_matrix, psi.localDim());

    auto node_i = psi.getTNodeRoot()->getItem(site_i);
    auto node_j = psi.getTNodeRoot()->getItem(site_j);

    // Print tensor dimensions at site_i and site_j before applying gate
    std::cout << "Tensor at site " << site_i << " dimensions: (" << node_i->getTensor().rows() << "x" << node_i->getTensor().cols() << ")" << std::endl;
    std::cout << "Tensor at site " << site_j << " dimensions: (" << node_j->getTensor().rows() << "x" << node_j->getTensor().cols() << ")" << std::endl;

    // Apply gate and reshape
    node_i->applyGateAndReshape(u_matrix);
    node_j->applyGateAndReshape(v_matrix);

    // Get the dimension of the gate
    int gate_dim = singular_values.size();

    // Wire all common nodes on the path from i to j (excluding the leaves)
    auto nodes = psi.getTNodeRoot()->getItem(site_i, site_j);
    for (auto& node : psi.getTNodeRoot()->getItem(site_i + 1, site_j - site_i - 1)) {
        node->update(gate_dim, site_i, site_j);
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
