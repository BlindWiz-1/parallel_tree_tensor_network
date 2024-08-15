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

const std::vector<std::shared_ptr<TNode>>& TNode::getChildren() const { // Add this method
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

void TNode::display(int depth) const {
    for (int i = 0; i < depth; ++i) {
        std::cout << "  ";
    }
    std::cout << name_ << std::endl;
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
    for (const auto& leaf : leaf_indices_) {
        if (std::stoi(leaf.first) == key) {
            return shared_from_this();
        }
    }
    for (const auto& child : children_) {
        auto result = child->getItem(key);
        if (result != nullptr) {
            return result;
        }
    }
    return nullptr;
}

std::vector<std::shared_ptr<TNode>> TNode::getItem(int start, int stop) {
    std::vector<std::shared_ptr<TNode>> result;
    auto start_node = getItem(start);
    auto stop_node = getItem(stop);
    if (start_node == nullptr || stop_node == nullptr) {
        return result;
    }

    Walker walker;
    auto path = walker.walk(start_node.get(), stop_node.get());
    for (const auto& node : path) {
        result.push_back(std::dynamic_pointer_cast<TNode>(node));
    }
    return result;
}
