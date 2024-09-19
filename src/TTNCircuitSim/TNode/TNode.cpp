#include "TNode.h"
#include <iostream>
#include <numeric>
#include "../../Operations/TTNContract/TTNContract.h"
#include "../../Operations/Walker/Walker.h"

TNode::TNode(const std::string& name, const Tensor& tensor, std::shared_ptr<TNode> parent)
    : name_(name), tensor_(tensor), local_dim_(tensor.rows()), parent_(parent), tmp_dim_(0), tmp_index_(-1) {}

void TNode::addChild(std::shared_ptr<TNode> child) {
    children_.push_back(child);
}

const Tensor& TNode::getTensor() const {
    return tensor_;
}

void TNode::setTensor(const Tensor& tensor) {
    tensor_ = tensor;
}

const std::string& TNode::getName() const {
    return name_;
}

const std::unordered_map<std::string, int>& TNode::getLeafIndices() const {
    return leaf_indices_;
}

void TNode::setLeafIndices(const std::unordered_map<std::string, int>& leaf_indices) {
    leaf_indices_ = leaf_indices;
}

bool TNode::isLeaf() const {
    return children_.empty();
}

bool TNode::isRoot() const {
    return parent_ == nullptr;
}

const std::vector<std::shared_ptr<TNode>>& TNode::getChildren() const {
    return children_;
}

int TNode::getTmpDim() const {
    return tmp_dim_;
}

void TNode::setTmpDim(int tmp_dim) {
    tmp_dim_ = tmp_dim;
}

int TNode::getTmpIndex() const {
    return tmp_index_;
}

void TNode::setTmpIndex(int tmp_index) {
    tmp_index_ = tmp_index;
}

std::optional<Tensor> TNode::getTmpFactor() const {
    return tmp_factor_;
}

void TNode::setTmpFactor(const std::optional<Tensor>& tmp_factor) {
    tmp_factor_ = tmp_factor;
}

std::shared_ptr<TNode> TNode::findRoot() {
    std::shared_ptr<TNode> current = shared_from_this();  // Start with the current node
    while (current->getParent() != nullptr) {  // Continue until there is no parent
        current = current->getParent();  // Move to the parent node
    }
    return current;  // Return the root node
}

void TNode::display(int depth) const {
    std::string indent = std::string(depth * 2, ' ');
    std::cout << indent << "|-- " << name_
                  << " (Dim: " << tensor_.rows() << "x" << tensor_.cols() << ", Children: " << children_.size() << ")"
                  << std::endl;

    for (const auto& child : children_) {
        child->display(depth + 1);
    }
}

std::shared_ptr<TNode> TNode::getParent() const {
    return parent_;
}

std::pair<double, int> TNode::countDimensions() const {
    auto dim_product = [](const Eigen::Index& rows, const Eigen::Index& cols) -> double {
        return static_cast<double>(rows * cols);
    };

    if (isLeaf()) {
        return {dim_product(tensor_.rows(), tensor_.cols()), static_cast<int>(std::max(tensor_.rows(), tensor_.cols()))};
    }

    double count = dim_product(tensor_.rows(), tensor_.cols());
    int current_max = static_cast<int>(std::max(tensor_.rows(), tensor_.cols()));

    for (const auto& child : children_) {
        auto [tmp_count, tmp_max] = child->countDimensions();
        count += tmp_count;
        current_max = std::max(current_max, tmp_max);
    }

    return {count, current_max};
}

void TNode::applyGate(const Tensor& gate_matrix) {
    assert(isLeaf());
    tensor_ = gate_matrix * tensor_;
}

void TNode::applyGateAndReshape(const Tensor& update) {
    assert(isLeaf());
    tensor_ = update * tensor_;
    tensor_.resize(local_dim_, tensor_.size() / local_dim_);
}

std::shared_ptr<TNode> TNode::getItem(int key) {
    Walker walker;
    auto root = this->findRoot();  // Method to find the root of the tree.

    auto path = walker.walk(root, nullptr);

    for (const auto& node : path) {
        if (node->getLeafIndices().find(std::to_string(key)) != node->getLeafIndices().end()) {
            return node;  // Return the node that matches the key.
        }
    }

    return nullptr;  // Return nullptr if no node with the key is found.
}

std::vector<std::shared_ptr<TNode>> TNode::getItem(int site_i, int site_j) {
    Walker walker;

    // Find the root node from the current node
    auto root = this->findRoot();

    // Use the simple getItem to find the starting and stopping nodes
    std::shared_ptr<TNode> node_i = root->getItem(site_i);
    std::shared_ptr<TNode> node_j = root->getItem(site_j);

    // If either of the nodes is null, return an empty result
    if (!node_i || !node_j) {
        return {};
    }

    // Use the walker to find the path between node_i and node_j
    auto path = walker.walk(node_i, node_j);

    return path;
}

void TNode::update(int gate_dim, int site_i, int site_j) {
    // Ensure this node is not a leaf
    assert(!isLeaf());

    // Determine the indices to update in the tensor dimensions
    int index_i = leaf_indices_.count(std::to_string(site_i)) ? leaf_indices_.at(std::to_string(site_i)) : tensor_.cols() - 1;
    int index_j = leaf_indices_.count(std::to_string(site_j)) ? leaf_indices_.at(std::to_string(site_j)) : tensor_.cols() - 1;

    // Get the current shape of the tensor
    Eigen::Index rows = tensor_.rows();
    Eigen::Index cols = tensor_.cols();

    // Create a new shape with the updated dimensions
    Eigen::Index new_rows = rows;
    Eigen::Index new_cols = cols;

    if (index_i < index_j) {
        new_rows *= gate_dim;
    } else {
        new_cols *= gate_dim;
    }

    Tensor new_tensor(new_rows, new_cols);
    new_tensor.block(0, 0, rows, cols) = tensor_;
    tensor_ = new_tensor;
}
