#include "PseudoTNode.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <cassert>
#include "../../Operations/TTNContract/TTNContract.h"
#include "../../Operations/Walker/Walker.h"

PseudoTNode::PseudoTNode(const std::string& name, const std::vector<int>& shape, std::shared_ptr<PseudoTNode> parent)
    : name_(name), shape_(shape), local_dim_(shape[0]), parent_(parent) {}

void PseudoTNode::addChild(std::shared_ptr<PseudoTNode> child) {
    children_.push_back(child);
}

const std::vector<int>& PseudoTNode::getShape() const {
    return shape_;
}

const std::string& PseudoTNode::getName() const {
    return name_;
}

const std::unordered_map<std::string, int>& PseudoTNode::getLeafIndices() const {
    return leaf_indices_;
}

void PseudoTNode::setLeafIndices(const std::unordered_map<std::string, int>& leaf_indices) {
    leaf_indices_ = leaf_indices;
}

bool PseudoTNode::isLeaf() const {
    return children_.empty();
}

bool PseudoTNode::isRoot() const {
    return parent_ == nullptr;
}

void PseudoTNode::display(int depth) const {
    for (int i = 0; i < depth; ++i) {
        std::cout << "  ";
    }
    std::cout << name_ << std::endl;
    for (const auto& child : children_) {
        child->display(depth + 1);
    }
}

std::shared_ptr<PseudoTNode> PseudoTNode::getParent() const {
    return parent_;
}

std::vector<std::shared_ptr<PseudoTNode>> PseudoTNode::getChildren() const {
    return children_;
}

void PseudoTNode::applyGate(const Tensor& gate_matrix) {
    assert(isLeaf());
    // In PseudoTNode, we just update the shape without actual tensor operations
}

void PseudoTNode::applyGateAndReshape(const Tensor& update) {
    assert(isLeaf());
    // In PseudoTNode, we just update the shape without actual tensor operations
    int gate_dim = update.cols();
    shape_ = {local_dim_, shape_[1] * gate_dim};
}

std::shared_ptr<PseudoTNode> PseudoTNode::getItem(int key) {
    if (std::stoi(name_) == key) {
        return shared_from_this();
    }
    for (const auto& child : children_) {
        auto result = child->getItem(key);
        if (result != nullptr) {
            return result;
        }
    }
    return nullptr;
}

std::vector<std::shared_ptr<PseudoTNode>> PseudoTNode::getItem(int start, int stop) {
    std::vector<std::shared_ptr<PseudoTNode>> result;
    auto start_node = getItem(start);
    auto stop_node = getItem(stop);
    if (start_node == nullptr || stop_node == nullptr) {
        return result;
    }

    Walker walker;
    auto path = walker.walk(start_node.get(), stop_node.get());
    for (const auto& node : path) {
        result.push_back(std::dynamic_pointer_cast<PseudoTNode>(node));
    }
    return result;
}
