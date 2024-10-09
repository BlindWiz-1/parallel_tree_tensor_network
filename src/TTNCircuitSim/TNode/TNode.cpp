#include "TNode.h"
#include <iostream>
#include <numeric>
#include "../../Operations/TTNContract/TTNContract.h"
#include "../../Operations/Walker/Walker.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <vector>

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
    tensor_ = gate_matrix * tensor_;
}

// Helper function: Converts Eigen::MatrixXcd to Eigen::Tensor
Eigen::Tensor<std::complex<double>, 2> matrixToTensor(const Eigen::MatrixXcd& matrix) {
    return Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 2>>(const_cast<std::complex<double>*>(matrix.data()), matrix.rows(), matrix.cols());
}

// Helper function: Converts Eigen::Tensor to Eigen::MatrixXcd
Eigen::MatrixXcd tensorToMatrix(const Eigen::Tensor<std::complex<double>, 2>& tensor) {
    return Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>>(tensor.data(), tensor.dimension(0), tensor.dimension(1));
}

// Helper function: Converts a 3D update matrix vector to an Eigen::Tensor
Eigen::Tensor<std::complex<double>, 3> convertUpdateToTensor(const std::vector<Eigen::MatrixXcd>& update) {
    int physical_dim = update[0].rows();
    int parent_dim = update[0].cols();
    int gate_dim = update.size();
    Eigen::Tensor<std::complex<double>, 3> update_tensor(physical_dim, parent_dim, gate_dim);
    for (int g = 0; g < gate_dim; ++g) {
        for (int p = 0; p < physical_dim; ++p) {
            for (int k = 0; k < parent_dim; ++k) {
                update_tensor(p, k, g) = update[g](p, k);
            }
        }
    }
    return update_tensor;
}

// Function to apply the gate and reshape tensor
void TNode::applyGateAndReshape(const std::vector<Eigen::MatrixXcd>& update) {
    // Log initial tensor dimensions
    std::cout << "Applying Gate and Reshape" << std::endl;
    std::cout << "Tensor dimensions (rows x cols): (" << tensor_.rows() << "x" << tensor_.cols() << ")" << std::endl;

    // Check that the update is not empty
    assert(!update.empty() && "Update tensor should not be empty!");

    int physical_dim = update[0].rows();
    int parent_dim = tensor_.cols();
    int gate_dim = update.size(); // Number of update matrices in the vector

    // Print update matrix dimensions for verification
    std::cout << "Physical dimension: " << physical_dim << std::endl;
    std::cout << "Parent dimension: " << parent_dim << std::endl;
    std::cout << "Gate dimension: " << gate_dim << std::endl;

    // Convert update matrices to a single Eigen::Tensor
    Eigen::Tensor<std::complex<double>, 3> update_tensor = convertUpdateToTensor(update);

    // Convert original tensor (Eigen::MatrixXcd) to Eigen::Tensor
    Eigen::Tensor<std::complex<double>, 2> tensor_eigen = matrixToTensor(tensor_);

    // Print tensor_eigen dimensions for verification
    std::cout << "Tensor to contract dimensions (rows x cols): (" << tensor_eigen.dimension(0) << "x" << tensor_eigen.dimension(1) << ")" << std::endl;

    // Perform the contraction operation: `np.einsum("abc, bd -> acd", update, self.tensor)`
    Eigen::array<Eigen::IndexPair<int>, 1> contraction_dims = {Eigen::IndexPair<int>(1, 0)};
    Eigen::Tensor<std::complex<double>, 3> result_tensor = update_tensor.contract(tensor_eigen, contraction_dims);

    // Print result_tensor dimensions after contraction
    std::cout << "Result tensor dimensions after contraction: ("
              << result_tensor.dimension(0) << "x"
              << result_tensor.dimension(1) << "x"
              << result_tensor.dimension(2) << ")" << std::endl;

    Eigen::Tensor<std::complex<double>, 2> reshaped_tensor = result_tensor.reshape(Eigen::array<Eigen::Index, 2>{physical_dim, parent_dim * gate_dim});
    std::cout << "Reshaped tensor dimensions: (" << reshaped_tensor.dimension(0) << "x" << reshaped_tensor.dimension(1) << ")" << std::endl;
    tensor_ = tensorToMatrix(reshaped_tensor);
    tensor_.resize(local_dim_, tensor_.size() / local_dim_);
    std::cout << "Final reshaped tensor dimensions: (" << tensor_.rows() << "x" << tensor_.cols() << ")" << std::endl;
}

std::shared_ptr<TNode> TNode::getItem(int key) {
    Walker walker;
    auto root = this->findRoot();

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

    if (isRoot()) {
        setTmpIndex(site_i);
        setTmpDim(gate_dim);
        return;
    }

    int index_i = leaf_indices_.count(std::to_string(site_i)) ? leaf_indices_.at(std::to_string(site_i)) : tensor_.cols() - 1;
    int index_j = leaf_indices_.count(std::to_string(site_j)) ? leaf_indices_.at(std::to_string(site_j)) : tensor_.cols() - 1;

    Eigen::Index rows = tensor_.rows();
    Eigen::Index cols = tensor_.cols();

    Eigen::Index new_rows = rows;
    Eigen::Index new_cols = cols;
    if (index_i < index_j) {
        new_rows *= gate_dim;
    } else {
        new_cols *= gate_dim;
    }

    Tensor new_tensor(new_rows, new_cols);
    new_tensor.block(0, 0, rows, cols) = tensor_;
    Eigen::MatrixXcd identity = Eigen::MatrixXcd::Identity(gate_dim, gate_dim);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < gate_dim; ++k) {
                if (index_i < index_j) {
                    new_tensor(i * gate_dim + k, j) = identity(k, k) * tensor_(i, j);
                } else {
                    new_tensor(i, j * gate_dim + k) = identity(k, k) * tensor_(i, j);
                }
            }
        }
    }

    tensor_ = new_tensor;
}
