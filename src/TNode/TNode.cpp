#include "TNode.h"
#include <iostream>
#include <numeric>

TNode::TNode(const std::string& name, const Tensor& tensor, std::shared_ptr<TNode> parent)
    : name_(name), tensor_(tensor), local_dim_(tensor.rows()), parent_(parent), tmp_dim_(0), tmp_index_(-1) {}

void TNode::addChild(std::shared_ptr<TNode> child) {
    children_.push_back(child);
}

const Tensor& TNode::getTensor() const {
    return tensor_;
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

const std::vector<std::shared_ptr<TNode>>& TNode::getChildren() const { // Add this method
    return children_;
}

void TNode::display(int depth) const {
    for (int i = 0; i < depth; ++i) {
        std::cout << "  ";
    }
    std::cout << name_ << std::endl;
    for (const auto& child : children_) {
        child->display(depth + 1);
    }
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
