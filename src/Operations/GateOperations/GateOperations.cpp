#include "GateOperations.h"
#include <cassert>
#include <iostream>
#include "../../TTNCircuitSim/TTN/TTN.h"

void GateOperations::applySingleParticleGate(TTN& psi, const Eigen::MatrixXcd& gate_matrix, int site) {
    assert(0 <= site && site < psi.nSites());
    psi.getTNodeRoot()->getItem(site)->applyGate(gate_matrix);
}


std::tuple<Tensor, Eigen::VectorXd, Tensor> GateOperations::decomposeTwoParticleGate(const Eigen::MatrixXcd& gate_matrix, int local_dimension) {

    // Reshape the gate matrix manually
    Eigen::MatrixXcd gate_matrix_reshaped(local_dimension * local_dimension, local_dimension * local_dimension);

    // Copy data into the reshaped matrix
    for (int i = 0; i < local_dimension; ++i) {
        for (int j = 0; j < local_dimension; ++j) {
            for (int k = 0; k < local_dimension; ++k) {
                for (int l = 0; l < local_dimension; ++l) {
                    gate_matrix_reshaped(i * local_dimension + j, k * local_dimension + l) =
                        gate_matrix(i * local_dimension + j, k * local_dimension + l);
                }
            }
        }
    }

    // Perform SVD on the reshaped gate matrix
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(gate_matrix_reshaped, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Tensor u_matrix = svd.matrixU();
    Tensor v_matrix = svd.matrixV().adjoint();  // Equivalent to transpose in complex space
    Eigen::VectorXd singular_values = svd.singularValues();

    // Filter small singular values
    Eigen::VectorXd::Index num_nonzero = (singular_values.array() > 1e-14).count();
    u_matrix = u_matrix.leftCols(num_nonzero);
    v_matrix = v_matrix.leftCols(num_nonzero);
    singular_values = singular_values.head(num_nonzero);

    // Reshape u_matrix and v_matrix to (local_dimension, local_dimension, -1)
    u_matrix.resize(local_dimension, local_dimension * num_nonzero);
    v_matrix.resize(local_dimension, local_dimension * num_nonzero);

    return std::make_tuple(u_matrix, singular_values, v_matrix);
}

void GateOperations::applyTwoParticleGate(TTN &psi, const Eigen::MatrixXcd &gate_matrix, int site_i, int site_j) {
    // Applies a two-particle gate to the tree tensor.
    assert(site_i >= 0 && site_i < site_j && site_j < psi.nSites());

    // Decompose the two-particle gate into U, singular values, and V
    Eigen::MatrixXcd u_matrix, v_matrix;
    Eigen::VectorXd singular_values;
    std::tie(u_matrix, singular_values, v_matrix) = decomposeTwoParticleGate(gate_matrix, psi.localDim());

    // Scale U and V by the square root of singular values
    u_matrix *= singular_values.array().sqrt().matrix().asDiagonal();
    v_matrix *= singular_values.array().sqrt().matrix().asDiagonal();

    // Update the leaves at site_i and site_j with U and V respectively
    auto node_i = psi.getTNodeRoot()->getItem(site_i);
    auto node_j = psi.getTNodeRoot()->getItem(site_j);
    node_i->applyGateAndReshape(u_matrix);
    node_j->applyGateAndReshape(v_matrix);

    // Get the dimension of the gate
    int gate_dim = singular_values.size();

    // Wire all common nodes on the path from i to j (excluding the leaves)
    auto nodes = psi.getTNodeRoot()->getItem(site_i, site_j);
    for (auto &node : psi.getTNodeRoot()->getItem(site_i + 1, site_j - site_i - 1)) {
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
            GateOperations::applySingleParticleGate(psi, gate.getGateMatrix(), gate.getSites()[0]);
        } else {
            GateOperations::applyTwoParticleGate(psi, gate.getGateMatrix(), gate.getSites()[0], gate.getSites()[1]);
            psi.orthonormalize(gate.getSites()[0], gate.getSites()[1]);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        if (duration.count() > 10) {
            std::cout << "Gate " << idx << " on " << gate.getSites()[0] << " " << gate.getSites()[1] << " took " << duration.count() << " seconds" << std::endl;
        } else {
            std::cout << "Gate " << idx << " on " << gate.getSites()[0] << " " << gate.getSites()[1] << " took " << duration.count() << " seconds" << std::endl;
        }
    }
}
