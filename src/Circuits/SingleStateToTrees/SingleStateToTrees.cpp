#include "SingleStateToTrees.h"
#include "../../TTNCircuitSim/TNode/TNode.h"
#include "../../TTNCircuitSim/PseudoTNode/PseudoTNode.h"
#include "../../Structure/SNode/SNode.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

// Helper function: Converts Eigen::MatrixXcd to Eigen::Tensor
Eigen::Tensor<std::complex<double>, 2> convertToTensor(const Eigen::MatrixXcd& matrix) {
    return Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 2>>(const_cast<std::complex<double>*>(matrix.data()), matrix.rows(), matrix.cols());
}

// Helper function: Converts Eigen::Tensor to Eigen::MatrixXcd
Eigen::MatrixXcd convertToMatrix(const Eigen::Tensor<std::complex<double>, 2>& tensor) {
    return Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>>(tensor.data(), tensor.dimension(0), tensor.dimension(1));
}

std::shared_ptr<TNode> singleStatesToTree(const std::vector<int>& single_states, int local_dimension, const std::shared_ptr<SNode>& structure) {
    return createTreeLevel(structure, single_states, local_dimension);
}

std::shared_ptr<PseudoTNode> createPseudoTree(const std::shared_ptr<SNode>& node, const std::vector<int>& single_states, int local_dimension) {
    return createPseudoTreeLevel(node, single_states, local_dimension);
}

std::shared_ptr<TNode> createTreeLevel(const std::shared_ptr<SNode>& node, const std::vector<int>& single_states, int local_dimension) {
    if (node->isLeaf()) { // Initialize leaf node with the single state value
        Tensor tensor(local_dimension, 1);
        tensor.setZero();
        tensor(single_states[std::stoi(node->getName())], 0) = 1;
        return std::make_shared<TNode>(node->getName(), tensor);
    }

    // For intermediate nodes, create a 3D tensor with dimensions 1x1x1
    Eigen::Tensor<std::complex<double>, 3> intermediate_tensor(1, 1, 1);
    intermediate_tensor.setConstant(1);  // Initialize to ones (or any other appropriate value)

    std::vector<std::shared_ptr<TNode>> child_nodes;
    std::unordered_map<std::string, int> leaf_indices;

    for (size_t idx = 0; idx < node->getChildren().size(); ++idx) {
        auto child = node->getChildren()[idx];
        auto new_child = createTreeLevel(child, single_states, local_dimension);

        if (new_child->isLeaf()) {
            leaf_indices[new_child->getName()] = idx;
        } else {
            for (const auto& [k, _] : new_child->getLeafIndices()) {
                leaf_indices[k] = idx;
            }
        }
        child_nodes.push_back(new_child);
    }

    // Create an intermediate node with a 1x1x1 tensor
    auto new_node = std::make_shared<TNode>(node->getName(), intermediate_tensor);
    for (const auto& child_node : child_nodes) {
        new_node->addChild(child_node);
    }

    new_node->setLeafIndices(leaf_indices);
    return new_node;
}

std::shared_ptr<PseudoTNode> createPseudoTreeLevel(const std::shared_ptr<SNode>& node, const std::vector<int>& single_states, int local_dimension) {
    if (node->isLeaf()) { // Adds base state for leaf nodes
        std::vector<int> shape = {local_dimension, 1};
        return std::make_shared<PseudoTNode>(node->getName(), shape);
    }

    // Intermediate nodes have dimensions 1x1x1
    std::vector<int> shape = {1, 1, 1}; // Adjust the shape to match the requirement
    std::vector<std::shared_ptr<PseudoTNode>> child_nodes;
    std::unordered_map<std::string, int> leaf_indices; // Dict to map child indices to leaves

    for (size_t idx = 0; idx < node->getChildren().size(); ++idx) {
        auto child = node->getChildren()[idx];
        auto new_child = createPseudoTreeLevel(child, single_states, local_dimension);
        if (new_child->isLeaf()) {
            leaf_indices[new_child->getName()] = idx;
        } else { // Add all entries of the child to own dict and set the index
            for (const auto& [k, _] : new_child->getLeafIndices()) {
                leaf_indices[k] = idx;
            }
        }
        child_nodes.push_back(new_child);
    }

    auto new_node = std::make_shared<PseudoTNode>(node->getName(), shape);
    for (const auto& child_node : child_nodes) {
        new_node->addChild(child_node);
    }

    new_node->setLeafIndices(leaf_indices);
    return new_node;
}
