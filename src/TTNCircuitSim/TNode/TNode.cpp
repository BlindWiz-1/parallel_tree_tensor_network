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


Tensor TNode::orthonormalizeQR(int i, const std::optional<Tensor>& factor) {
    Tensor tensor = tensor_;
    std::string index_str = std::to_string(i);
    if (factor.has_value()) {
        tensor = contractFactorOnIndex(tensor, factor.value(), leaf_indices_[index_str]);
    }
    Eigen::Index rows = tensor.rows();
    Eigen::Index cols = tensor.cols();
    Eigen::MatrixXd real_tensor = tensor.real();  // QR is typically done on real matrices
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(real_tensor);
    Eigen::MatrixXd q_matrix = qr.householderQ() * Eigen::MatrixXd::Identity(rows, cols);
    Eigen::MatrixXd r_matrix = qr.matrixQR().triangularView<Eigen::Upper>();
    tensor_ = q_matrix.cast<std::complex<double>>();
    return r_matrix.cast<std::complex<double>>();
}

Tensor TNode::orthonormalizeSVD(int i, double tol, int d_max, const std::optional<Tensor>& factor) {
    Tensor tensor = tensor_;
    if (factor.has_value()) {
        tensor = contractFactorOnIndex(tensor, factor.value(), leaf_indices_.at(std::to_string(i)));
    }
    Eigen::JacobiSVD<Tensor> svd(tensor, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Tensor u_matrix = svd.matrixU();
    Tensor singular_values = svd.singularValues();
    Tensor v_matrix = svd.matrixV();
    int effective_d_max = std::min(d_max, static_cast<int>(singular_values.size()));
    tensor_ = u_matrix.leftCols(effective_d_max);
    return (singular_values.topRows(effective_d_max).asDiagonal() * v_matrix.leftCols(effective_d_max).adjoint());
}

void TNode::precontractRoot(int site_j, const Tensor& factor) {
    assert(isRoot());
    Tensor tensor = tensor_;
    int site_i = leaf_indices_.at(std::to_string(tmp_index_));
    site_j = leaf_indices_.at(std::to_string(site_j));

    Eigen::Index rows = tmp_factor_.value().rows();
    Eigen::Index cols = tmp_factor_.value().cols() / tmp_dim_;
    Tensor reshaped_tmp_factor(rows, tmp_dim_ * cols);
    reshaped_tmp_factor = tmp_factor_.value().reshaped(rows, tmp_dim_ * cols);

    rows = factor.rows();
    cols = factor.cols() / tmp_dim_;
    Tensor reshaped_factor(rows, tmp_dim_ * cols);
    reshaped_factor = factor.reshaped(rows, tmp_dim_ * cols);

    tensor_ = contractFactorOnIndex(tensor, reshaped_tmp_factor, site_i);
    tensor_ = contractFactorOnIndex(tensor_, reshaped_factor, site_j);

    tmp_dim_ = 0;
    tmp_factor_.reset();
    tmp_index_ = -1;
}

Tensor TNode::contractFactorOnIndex(const Tensor& tensor, const Tensor& factor, int idx) const {
    Eigen::Index rows = tensor.rows();
    Eigen::Index cols = tensor.cols();
    Tensor result = tensor;
    for (Eigen::Index i = 0; i < rows; ++i) {
        for (Eigen::Index j = 0; j < cols; ++j) {
            result(i, j) *= factor(i % factor.rows(), j % factor.cols());
        }
    }
    return result;
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
