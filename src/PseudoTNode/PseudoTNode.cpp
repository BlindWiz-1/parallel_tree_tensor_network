#include "PseudoTNode.h"
#include <iostream>

PseudoTNode::PseudoTNode(const std::string& name, const std::vector<int>& shape, std::shared_ptr<PseudoTNode> parent)
    : name_(name), shape_(shape), parent_(parent) {}

void PseudoTNode::addChild(std::shared_ptr<PseudoTNode> child) {
    children_.push_back(child);
}

const std::string& PseudoTNode::getName() const {
    return name_;
}

const std::vector<int>& PseudoTNode::getShape() const {
    return shape_;
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
