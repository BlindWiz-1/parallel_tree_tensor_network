#include "FindTreeStructure.h"
#include "../QPair/QPair.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

// Function to calculate Euclidean distance between two vectors
double calculateDistance(const Eigen::VectorXd& point1, const Eigen::VectorXd& point2) {
    return (point1 - point2).norm();
}

// K-means clustering implementation using Eigen
std::vector<int> kMeansClustering(const Eigen::MatrixXd& similarity, int clusters, int maxIterations = 100) {
    int n = similarity.rows();   // Number of data points
    int d = similarity.cols();   // Dimension of each data point

    Eigen::MatrixXd centroids(clusters, d);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n - 1);
    for (int i = 0; i < clusters; ++i) {
        centroids.row(i) = similarity.row(dis(gen));
    }

    std::vector<int> labels(n, 0);

    for (int iter = 0; iter < maxIterations; ++iter) {
        bool isConverged = true;

        for (int i = 0; i < n; ++i) {
            double minDistance = std::numeric_limits<double>::max();
            int closestCentroid = 0;

            for (int j = 0; j < clusters; ++j) {
                double distance = calculateDistance(similarity.row(i), centroids.row(j));
                if (distance < minDistance) {
                    minDistance = distance;
                    closestCentroid = j;
                }
            }

            // Check if the label has changed
            if (labels[i] != closestCentroid) {
                isConverged = false;
                labels[i] = closestCentroid;
            }
        }

        if (isConverged) {
            break;
        }

        centroids = Eigen::MatrixXd::Zero(clusters, d);
        Eigen::VectorXi count = Eigen::VectorXi::Zero(clusters);

        for (int i = 0; i < n; ++i) {
            centroids.row(labels[i]) += similarity.row(i);
            count(labels[i])++;
        }

        for (int j = 0; j < clusters; ++j) {
            if (count(j) > 0) {
                centroids.row(j) /= count(j);
            }
        }
    }

    return labels;
}

std::shared_ptr<SNode> findTreeStructure(const Circuit& circuit, int random_state, bool flat) {
    Eigen::MatrixXd similarity = toSimilarityMatrix(circuit);

    int clusters = std::ceil(static_cast<double>(circuit.getLSites()) / 8);
    std::vector<int> labels = kMeansClustering(similarity, clusters);

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

    for (const CircuitGate& gate : circuit.getGates()) {
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
    for (const QPair& entry : entries) {
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
