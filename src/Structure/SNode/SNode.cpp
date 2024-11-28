#include "SNode.h"
#include <iostream>

SNode::SNode(const std::string& name, std::shared_ptr<SNode> parent, const std::vector<std::shared_ptr<SNode>>& children)
    : name_(name), parent_(parent), children_(children) {}

void SNode::addChild(std::shared_ptr<SNode> child) {
    child->parent_ = shared_from_this();  // Set this node as the parent of the child
    children_.push_back(child);
}

const std::string& SNode::getName() const {
    return name_;
}

void SNode::setName(const std::string& name) {
    name_ = name;
}

std::shared_ptr<SNode> SNode::getParent() const {
    return parent_;
}

const std::vector<std::shared_ptr<SNode>>& SNode::getChildren() const {
    return children_;
}

bool SNode::isLeaf() const {
    return children_.empty();
}

bool SNode::isRoot() const {
    return parent_ == nullptr;
}

void SNode::display(int depth) const {
    for (int i = 0; i < depth; ++i) {
        std::cout << "  ";
    }
    std::cout << name_ << std::endl;

    for (const auto& child : children_) {
        child->display(depth + 1);
    }
}
