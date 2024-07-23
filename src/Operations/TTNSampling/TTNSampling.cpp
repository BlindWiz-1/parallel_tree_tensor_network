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
        Tensor tensor = node->getTensor();
        tensor = tensor * current_state * tensor.adjoint();
        int state = sampleQubit(tensor);
        node->getTensor().col(state);
        return {state};
    }
    std::vector<int> states;
    for (const auto& child : node->getChildren()) {
        auto child_states = sampleAndContract(child, current_state);
        states.insert(states.end(), child_states.begin(), child_states.end());
        Tensor child_tensor = child->getTensor();
        current_state = current_state * child_tensor * child_tensor.adjoint();
    }
    node->setTensor(contract(node, 1.0));
    return states;
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
