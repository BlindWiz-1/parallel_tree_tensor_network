#include "FindTreeStructure.h"
#include "../QPair/QPair.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Eigenvalues>

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

// K-means clustering to take an external generator for random initialization
std::vector<int> kMeansClustering(const Eigen::MatrixXd& data, int clusters, std::mt19937& gen, int maxIterations = 100) {
    int n = data.rows();
    int d = data.cols();

    Eigen::MatrixXd centroids(clusters, d);
    std::uniform_int_distribution<> dis(0, n - 1);
    for (int i = 0; i < clusters; ++i) {
        centroids.row(i) = data.row(dis(gen));
    }

    std::vector<int> labels(n, 0);
    for (int iter = 0; iter < maxIterations; ++iter) {
        bool isConverged = true;
        for (int i = 0; i < n; ++i) {
            double minDistance = std::numeric_limits<double>::max();
            int closestCentroid = 0;
            for (int j = 0; j < clusters; ++j) {
                double distance = calculateDistance(data.row(i), centroids.row(j));
                if (distance < minDistance) {
                    minDistance = distance;
                    closestCentroid = j;
                }
            }

            if (labels[i] != closestCentroid) {
                isConverged = false;
                labels[i] = closestCentroid;
            }
        }

        if (isConverged) break;

        centroids = Eigen::MatrixXd::Zero(clusters, d);
        Eigen::VectorXi count = Eigen::VectorXi::Zero(clusters);
        for (int i = 0; i < n; ++i) {
            centroids.row(labels[i]) += data.row(i);
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

// Spectral clustering implementation
std::vector<int> spectralClustering(const Eigen::MatrixXd& similarity, int clusters, int random_state) {
    int n = similarity.rows();

    Eigen::MatrixXd degree = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        degree(i, i) = similarity.row(i).sum();
    }

    Eigen::MatrixXd degree_inv_sqrt = (degree.array() + 1e-9).inverse().sqrt().matrix();
    Eigen::MatrixXd laplacian = Eigen::MatrixXd::Identity(n, n) - degree_inv_sqrt * similarity * degree_inv_sqrt;

    if (laplacian.hasNaN()) {
        std::cerr << "Error: Laplacian contains NaN values." << std::endl;
        return {};
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(laplacian);
    if (eigensolver.info() != Eigen::Success) {
        std::cerr << "Error: Eigenvalue decomposition failed." << std::endl;
        return {};
    }

    Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors().leftCols(clusters);

    std::mt19937 gen(random_state);
    auto labels = kMeansClustering(eigenvectors, clusters, gen);
    return labels;
}

std::shared_ptr<SNode> findTreeStructure(const Circuit& circuit, int random_state, bool flat) {
    Eigen::MatrixXd similarity = toSimilarityMatrix(circuit);
    int clusters = std::max(2, (int)std::ceil(static_cast<double>(circuit.getLSites()) / 8));

    // Use spectral clustering with random_state for consistent clustering results
    std::vector<int> labels = spectralClustering(similarity, clusters, random_state);

    // Group leaves by clusters
    std::vector<std::vector<int>> cluster_groups(clusters);
    for (int i = 0; i < labels.size(); ++i) {
        cluster_groups[labels[i]].push_back(i);
    }

    std::vector<std::shared_ptr<SNode>> children;
    for (int i = 0; i < clusters; ++i) {
        auto subtree = flat ? createSubtreeFlat(cluster_groups[i], i) : createSubtree(cluster_groups[i], similarity, i);
        if (subtree) {
            children.push_back(subtree);
        }
    }

    return std::make_shared<SNode>("root", nullptr, children);
}

// Function to compute the similarity matrix for the circuit
Eigen::MatrixXd toSimilarityMatrix(const Circuit& circuit) {
    int n = circuit.getLSites();
    Eigen::MatrixXd similarity = Eigen::MatrixXd::Zero(n, n);

    std::unordered_map<std::pair<int, int>, int, pair_hash> pairwise_gate_count;
    std::unordered_map<int, int> total_gate_count;

    for (const CircuitGate& gate : circuit.getGates()) {
        const auto& sites = gate.getSites();
        if (sites.size() == 1) {
            total_gate_count[sites[0]] += gate.getGateMatrix().cols();
        } else if (sites.size() == 2) {
            int q1 = sites[0], q2 = sites[1];
            pairwise_gate_count[{q1, q2}] += gate.getGateMatrix().cols();
            total_gate_count[q1] += gate.getGateMatrix().cols();
            total_gate_count[q2] += gate.getGateMatrix().cols();
        }
    }

    for (const auto& [pair, count] : pairwise_gate_count) {
        int q1 = pair.first, q2 = pair.second;
        double sim = count + 1.0 / (total_gate_count[q1] + total_gate_count[q2] + 1e-9);
        similarity(q1, q2) = sim;
        similarity(q2, q1) = sim;
    }

    for (int i = 0; i < n; ++i) {
        similarity(i, i) = total_gate_count[i];
    }
    similarity /= similarity.maxCoeff(); // Normalize similarity matrix
    return similarity;
}

std::shared_ptr<SNode> createSubtree(const std::vector<int>& leaves, const Eigen::MatrixXd& similarity, int cluster) {
    if (leaves.empty()) {
        return nullptr;
    }

    // If only one leaf, return it directly
    if (leaves.size() == 1) {
        return std::make_shared<SNode>(std::to_string(leaves[0]));
    }

    // Create nodes for each leaf
    std::vector<std::shared_ptr<SNode>> node_vector;
    for (int leaf : leaves) {
        node_vector.push_back(std::make_shared<SNode>(std::to_string(leaf)));
    }

    int node_counter = leaves.size(); // Counter for unique intermediate nodes

    // Combine nodes into a tree structure
    while (node_vector.size() > 1) {
        double max_similarity = -1.0;
        size_t best_left_idx = 0, best_right_idx = 0;

        // Find the pair of nodes with the highest similarity
        for (size_t i = 0; i < node_vector.size(); ++i) {
            for (size_t j = i + 1; j < node_vector.size(); ++j) {
                int left_id = std::stoi(node_vector[i]->getName());
                int right_id = std::stoi(node_vector[j]->getName());
                double sim = similarity(left_id, right_id);

                if (sim > max_similarity) {
                    max_similarity = sim;
                    best_left_idx = i;
                    best_right_idx = j;
                }
            }
        }

        // Create a new parent node for the most similar pair
        auto parent = std::make_shared<SNode>(std::to_string(cluster) + "." + std::to_string(node_counter++));
        parent->addChild(node_vector[best_left_idx]);
        parent->addChild(node_vector[best_right_idx]);

        // Remove the paired nodes and add the parent node
        if (best_left_idx > best_right_idx) std::swap(best_left_idx, best_right_idx); // Ensure indices are in order
        node_vector.erase(node_vector.begin() + best_right_idx);
        node_vector.erase(node_vector.begin() + best_left_idx);
        node_vector.push_back(parent);
    }

    // Return the root of the subtree
    return node_vector.front();
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

    // Create intermediate nodes for every pair of leaves
    std::queue<std::shared_ptr<SNode>> node_queue;
    for (int leaf : leaves) {
        node_queue.push(std::make_shared<SNode>(std::to_string(leaf)));
    }

    while (node_queue.size() > 1) {
        auto left = node_queue.front();
        node_queue.pop();
        auto right = node_queue.front();
        node_queue.pop();
        auto parent = std::make_shared<SNode>(std::to_string(cluster) + "." + std::to_string(leaves.size()));
        parent->addChild(left);
        parent->addChild(right);
        node_queue.push(parent);
    }

    return node_queue.front();
}
