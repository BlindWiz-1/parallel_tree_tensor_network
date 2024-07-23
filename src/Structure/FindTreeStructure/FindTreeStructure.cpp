#include "FindTreeStructure.h"
#include "../QPair/QPair.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>

// Hash function for std::pair
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

std::shared_ptr<SNode> findTreeStructure(const Circuit& circuit, int clusters, int random_state, int d_max, bool flat) {
    if (circuit.getLSites() == 1) {
        return std::make_shared<SNode>("0");
    }

    if (clusters == -1) {
        clusters = std::ceil(static_cast<double>(circuit.getLSites()) / 8);
    }
    if (d_max == -1) {
        d_max = std::ceil(static_cast<double>(circuit.getLSites()) / clusters);
    }

    Eigen::MatrixXd similarity = toSimilarityMatrix(circuit);
    std::vector<int> labels(circuit.getLSites(), 0);

    // Perform clustering (using KMeans or any other clustering algorithm)
    // Here, we use a simple example with random clustering
    std::default_random_engine generator(random_state);
    std::uniform_int_distribution<int> distribution(0, clusters - 1);
    for (int i = 0; i < circuit.getLSites(); ++i) {
        labels[i] = distribution(generator);
    }

    std::vector<std::shared_ptr<SNode>> children;
    for (int i = 0; i < clusters; ++i) {
        std::vector<int> leaves;
        for (int j = 0; j < labels.size(); ++j) {
            if (labels[j] == i) {
                leaves.push_back(j);
            }
        }
        std::shared_ptr<SNode> subtree = flat ? createSubtreeFlat(leaves, i) : createSubtree(leaves, similarity, i);
        if (subtree) {
            children.push_back(subtree);
        }
    }

    return std::make_shared<SNode>("root", nullptr, children);
}

Eigen::MatrixXd toSimilarityMatrix(const Circuit& circuit) {
    int n = circuit.getLSites();
    Eigen::MatrixXd similarity = Eigen::MatrixXd::Identity(n, n) * std::pow(2, 63);
    std::unordered_map<std::pair<int, int>, int, pair_hash> pairwise_gate_count;
    std::unordered_map<int, int> total_gate_count;

    for (const auto& gate : circuit.getGates()) {
        if (gate.getSites().size() == 2) {
            int q1 = gate.getSites()[0];
            int q2 = gate.getSites()[1];
            pairwise_gate_count[{q1, q2}] += gate.getGateMatrix().cols();
            total_gate_count[q1] += gate.getGateMatrix().cols();
            total_gate_count[q2] += gate.getGateMatrix().cols();
        }
    }

    for (const auto& [pair, count] : pairwise_gate_count) {
        int q1 = pair.first;
        int q2 = pair.second;
        double sim = count + 1.0 / (total_gate_count[q1] + total_gate_count[q2]);
        similarity(q1, q2) = sim;
        similarity(q2, q1) = sim;
    }

    return similarity;
}

std::shared_ptr<SNode> createSubtree(const std::vector<int>& leaves, const Eigen::MatrixXd& similarity, int cluster) {
    if (leaves.empty()) {
        return nullptr;
    }
    if (leaves.size() == 1) {
        return std::make_shared<SNode>(std::to_string(leaves[0]));
    }

    std::vector<QPair> entries;
    for (size_t i = 0; i < leaves.size(); ++i) {
        for (size_t j = i + 1; j < leaves.size(); ++j) {
            entries.emplace_back(std::make_pair(leaves[i], leaves[j]), similarity(leaves[i], leaves[j]));
        }
    }

    std::sort(entries.begin(), entries.end());
    std::unordered_set<int> seen;
    int counter = std::count_if(entries.begin(), entries.end(), [](const QPair& entry) { return entry.getSimilarity() > 0; }) - 1;
    auto current = std::make_shared<SNode>(std::to_string(cluster) + "." + std::to_string(counter));

    double sim = entries[0].getSimilarity();
    for (const auto& entry : entries) {
        int i = entry.getQubits().first;
        int j = entry.getQubits().second;
        std::vector<std::shared_ptr<SNode>> new_leaves;

        if (seen.find(i) == seen.end()) {
            new_leaves.push_back(std::make_shared<SNode>(std::to_string(i)));
        }
        if (seen.find(j) == seen.end()) {
            new_leaves.push_back(std::make_shared<SNode>(std::to_string(j)));
        }
        if (new_leaves.empty()) {
            continue;
        }
        if (sim == entry.getSimilarity() && (new_leaves.size() == 1 || current->getChildren().empty())) {
            for (const auto& leaf : new_leaves) {
                current->addChild(leaf);
            }
        } else {
            --counter;
            auto new_node = std::make_shared<SNode>(std::to_string(cluster) + "." + std::to_string(counter));
            new_node->addChild(current);
            for (const auto& leaf : new_leaves) {
                new_node->addChild(leaf);
            }
            current = new_node;
        }
        seen.insert(i);
        seen.insert(j);
        sim = entry.getSimilarity();
    }

    return current;
}

int maxClusterSize(const std::vector<int>& labels) {
    std::unordered_map<int, int> count;
    for (int label : labels) {
        ++count[label];
    }
    int max_size = 0;
    for (const auto& [key, value] : count) {
        if (value > max_size) {
            max_size = value;
        }
    }
    return max_size;
}

std::shared_ptr<SNode> createSubtreeFlat(const std::vector<int>& leaves, int cluster) {
    if (leaves.empty()) {
        return nullptr;
    }
    if (leaves.size() == 1) {
        return std::make_shared<SNode>(std::to_string(leaves[0]));
    }
    std::vector<std::shared_ptr<SNode>> children;
    for (int leave : leaves) {
        children.push_back(std::make_shared<SNode>(std::to_string(leave)));
    }
    return std::make_shared<SNode>(std::to_string(cluster) + ".0", nullptr, children);
}
