//
// Created by Denado Rabeli on 7/19/24.
//
#include "TTNSampling.h"
#include <numeric>
#include <random>
#include <algorithm>
#include <iostream>
#include "../TTNContract/TTNContract.h"
#include "../../TTNCircuitSim/TNode/TNode.h"

static std::vector<Tensor> BASE_OP = {
    (Tensor(2, 2) << 1, 0, 0, 0).finished(),  // |0>
    (Tensor(2, 2) << 0, 0, 0, 1).finished()   // |1>
};

Tensor TTNSampling::sample(std::shared_ptr<TNode> root, double nrm) {
    Tensor current_state = Tensor::Identity(2, 2) * std::abs(nrm) * std::abs(nrm);
    auto state_sample = sampleAndContract(root, current_state);
    Eigen::VectorXi result = Eigen::Map<Eigen::VectorXi>(state_sample.data(), state_sample.size());
    return result.cast<std::complex<double>>();
}

std::vector<int> TTNSampling::sampleAndContract(std::shared_ptr<TNode> node, Tensor& current_state) {
    if (node->isLeaf()) {
        auto tensor_variant = node->getTensor();
        if (std::holds_alternative<Eigen::MatrixXcd>(tensor_variant)) {
            // Leaf nodes should have Eigen::MatrixXcd tensors
            Tensor tensor = std::get<Eigen::MatrixXcd>(tensor_variant);

            // Compute the new current state
            Tensor new_current_state = tensor * current_state * tensor.adjoint();

            // Sample the qubit
            int state = sampleQubit(new_current_state);

            // Update the node's tensor to the selected state (column)
            Tensor selected_state = tensor.col(state);
            node->setTensor(selected_state);

            return {state};
        } else {
            throw std::runtime_error("Leaf node does not have a MatrixXcd tensor.");
        }
    } else {
        std::vector<int> states;
        for (const auto& child : node->getChildren()) {
            auto child_states = sampleAndContract(child, current_state);
            states.insert(states.end(), child_states.begin(), child_states.end());

            auto child_tensor_variant = child->getTensor();
            if (std::holds_alternative<Eigen::MatrixXcd>(child_tensor_variant)) {
                Tensor child_tensor = std::get<Eigen::MatrixXcd>(child_tensor_variant);
                // Update current_state with the child's tensor
                current_state = current_state * child_tensor * child_tensor.adjoint();
            } else if (std::holds_alternative<Eigen::Tensor<std::complex<double>, 3>>(child_tensor_variant)) {
                // Handle Eigen::Tensor<std::complex<double>, 3>
                Eigen::Tensor<std::complex<double>, 3> child_tensor = std::get<Eigen::Tensor<std::complex<double>, 3>>(child_tensor_variant);

                // Flatten the 3D tensor to a 2D matrix for multiplication
                Eigen::Index dim0 = child_tensor.dimension(0);
                Eigen::Index dim1 = child_tensor.dimension(1);
                Eigen::Index dim2 = child_tensor.dimension(2);

                Eigen::MatrixXcd child_matrix(dim0 * dim1, dim2);
                const std::complex<double>* data_ptr = child_tensor.data();
                for (Eigen::Index idx = 0; idx < child_matrix.size(); ++idx) {
                    child_matrix(idx) = data_ptr[idx];
                }

                // Update current_state
                current_state = current_state * child_matrix * child_matrix.adjoint();
            } else {
                throw std::runtime_error("Unknown tensor type in child node.");
            }
        }

        // After processing children, contract the node
        auto contracted_tensor = contract(node, 1.0);  // Adjusted to return std::variant

        // Set the node's tensor
        node->setTensor(contracted_tensor);

        return states;
    }
}

int TTNSampling::sampleQubit(const Tensor& tensor, int shots) {
    std::vector<double> state_probabilities(BASE_OP.size());
    for (size_t i = 0; i < BASE_OP.size(); ++i) {
        state_probabilities[i] = (tensor * BASE_OP[i]).trace().real();
    }
    double total_probability = std::accumulate(state_probabilities.begin(), state_probabilities.end(), 0.0);
    for (auto& prob : state_probabilities) {
        prob /= total_probability;
    }

    std::vector<int> results(2, 0);
    for (int i = 0; i < shots; ++i) {
        int idx = sampleOnce(state_probabilities);
        results[idx]++;
    }
    return std::distance(results.begin(), std::max_element(results.begin(), results.end()));
}

int TTNSampling::sampleOnce(const std::vector<double>& state_probabilities) {
    std::vector<double> prefix_sum(state_probabilities.size());
    std::partial_sum(state_probabilities.begin(), state_probabilities.end(), prefix_sum.begin());
    double val = static_cast<double>(rand()) / RAND_MAX;
    return std::distance(prefix_sum.begin(), std::lower_bound(prefix_sum.begin(), prefix_sum.end(), val));
}
