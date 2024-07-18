#include "SingleStateToTrees.h"
#include <Eigen/Dense>

std::shared_ptr<TNode> singleStatesToTree(const std::vector<int>& single_states, int local_dimension, const std::shared_ptr<SNode>& structure) {
    return createTreeLevel(structure, single_states, local_dimension);
}

std::shared_ptr<PseudoTNode> createPseudoTree(const std::shared_ptr<SNode>& node, const std::vector<int>& single_states, int local_dimension) {
    return createPseudoTreeLevel(node, single_states, local_dimension);
}

std::shared_ptr<TNode> createTreeLevel(const std::shared_ptr<SNode>& node, const std::vector<int>& single_states, int local_dimension) {
    if (node->isLeaf()) { // adds base state for leaf nodes
        Tensor tensor(local_dimension, 1);
        tensor.setZero();
        tensor(single_states[std::stoi(node->getName())], 0) = 1;
        return std::make_shared<TNode>(node->getName(), tensor);
    }

    // one index per child + one for the parent node
    int num_indices = node->getChildren().size() + 1;
    Tensor tensor = Tensor::Ones(1, 1);

    // Resize tensor to the new shape
    tensor.resize(Eigen::Dynamic, Eigen::Dynamic);
    tensor.resize(num_indices, 1);

    std::vector<std::shared_ptr<TNode>> child_nodes;
    std::unordered_map<std::string, int> leaf_indices; // dict to map child indices to leaves

    for (size_t idx = 0; idx < node->getChildren().size(); ++idx) {
        auto child = node->getChildren()[idx];
        auto new_child = createTreeLevel(child, single_states, local_dimension);
        if (new_child->isLeaf()) {
            leaf_indices[new_child->getName()] = idx;
        } else { // add all entries of the child to own dict and set the index
            for (const auto& [k, _] : new_child->getLeafIndices()) {
                leaf_indices[k] = idx;
            }
        }
        child_nodes.push_back(new_child);
    }

    auto new_node = std::make_shared<TNode>(node->getName(), tensor);
    for (const auto& child_node : child_nodes) {
        new_node->addChild(child_node);
    }

    new_node->setLeafIndices(leaf_indices);
    return new_node;
}

std::shared_ptr<PseudoTNode> createPseudoTreeLevel(const std::shared_ptr<SNode>& node, const std::vector<int>& single_states, int local_dimension) {
    if (node->isLeaf()) { // adds base state for leaf nodes
        std::vector<int> shape = {local_dimension, 1};
        return std::make_shared<PseudoTNode>(node->getName(), shape);
    }

    // one index per child + one for the parent node
    std::vector<int> shape(node->getChildren().size() + 1, 1);
    std::vector<std::shared_ptr<PseudoTNode>> child_nodes;
    std::unordered_map<std::string, int> leaf_indices; // dict to map child indices to leaves

    for (size_t idx = 0; idx < node->getChildren().size(); ++idx) {
        auto child = node->getChildren()[idx];
        auto new_child = createPseudoTreeLevel(child, single_states, local_dimension);
        if (new_child->isLeaf()) {
            leaf_indices[new_child->getName()] = idx;
        } else { // add all entries of the child to own dict and set the index
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
