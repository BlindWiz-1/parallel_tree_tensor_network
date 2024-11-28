#include "TNode.h"
#include "../../Operations/Walker/Walker.h"
#include <iostream>
#include <queue>
#include <vector>
#include <cassert>
#include <set>

TNode::TNode(const std::string& name, const Tensor& tensor, std::shared_ptr<TNode> parent)
    : name_(name), tensor_matrix_(tensor), parent_(parent), local_dim_(tensor.rows()), tmp_dim_(0), tmp_index_(-1) {}

TNode::TNode(const std::string& name, const Eigen::Tensor<std::complex<double>, 3>& tensor, std::shared_ptr<TNode> parent)
    : name_(name), tensor_tensor_(tensor), parent_(parent), local_dim_(tensor.dimension(0)), tmp_dim_(0), tmp_index_(-1) {}

void TNode::addChild(std::shared_ptr<TNode> child) {
    child->parent_ = shared_from_this();  // Set this node as the parent of the child
    children_.push_back(child);
}

std::variant<Tensor, Eigen::Tensor<std::complex<double>, 3>> TNode::getTensor() const {
    if (tensor_matrix_) {
        return tensor_matrix_.value();
    } else if (tensor_tensor_) {
        return tensor_tensor_.value();
    } else {
        throw std::runtime_error("Tensor data is not set.");
    }
}

void TNode::setTensor(const Tensor& tensor) {
    tensor_matrix_ = tensor;
    tensor_tensor_.reset();
}

void TNode::setTensor(const Eigen::Tensor<std::complex<double>, 3>& tensor) {
    tensor_tensor_ = tensor;
    tensor_matrix_.reset();
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
    auto current = shared_from_this();
    while (current->getParent()) {
        current = current->getParent();
    }
    return current;
}

void TNode::display(int depth) const {
    std::string indent(depth * 2, ' ');
    std::cout << indent << "|-- " << name_ << " (Dim: ";

    // Display dimensions
    auto dims = getTensorDimensions();
    for (size_t i = 0; i < dims.size(); ++i) {
        std::cout << dims[i];
        if (i < dims.size() - 1) {
            std::cout << "x";
        }
    }

    // Display parent information
    if (parent_) {
        std::cout << ", Parent: " << parent_->getName();
    } else {
        std::cout << ", Parent: None (Root)";
    }

    // Display number of children
    std::cout << ", Children: " << children_.size() << ")" << std::endl;

    // Recursively display children
    for (const auto& child : children_) {
        child->display(depth + 1);
    }
}

std::shared_ptr<TNode> TNode::getParent() const {
    return parent_;
}

std::pair<double, int> TNode::countDimensions() const {
    double count = 0;
    int max_dim = 0;
    auto dims = getTensorDimensions();
    double size = 1;
    for (int dim : dims) {
        size *= dim;
        if (dim > max_dim) {
            max_dim = dim;
        }
    }
    count += size;

    for (const auto& child : children_) {
        auto [child_count, child_max_dim] = child->countDimensions();
        count += child_count;
        if (child_max_dim > max_dim) {
            max_dim = child_max_dim;
        }
    }
    return {count, max_dim};
}

void TNode::applyGate(const Tensor& gate_matrix) {
    assert(isLeaf() && "applyGate should only be called on leaf nodes");
    tensor_matrix_ = gate_matrix * tensor_matrix_.value();
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

void TNode::applyGateAndReshape(const std::vector<Eigen::MatrixXcd>& update) {
    assert(isLeaf() && "Only applied to leaf");
    assert(!update.empty() && "Update tensor should not be empty!");
    assert(tensor_matrix_.has_value() && "Tensor matrix is not set for this leaf node.");

    int physical_dim = update[0].rows();
    int parent_dim = tensor_matrix_->cols();
    int gate_dim = static_cast<int>(update.size()); // Number of update matrices in the vector

    // Print update matrix dimensions for verification
    std::cout << "Physical dimension: " << physical_dim << std::endl;
    std::cout << "Parent dimension: " << parent_dim << std::endl;
    std::cout << "Gate dimension: " << gate_dim << std::endl;

    // Convert update matrices to a single Eigen::Tensor
    Eigen::Tensor<std::complex<double>, 3> update_tensor = convertUpdateToTensor(update);

    // Convert original tensor (Eigen::MatrixXcd) to Eigen::Tensor
    Eigen::Tensor<std::complex<double>, 2> tensor_eigen = matrixToTensor(*tensor_matrix_);

    // Print tensor_eigen dimensions for verification
    std::cout << "Tensor to contract dimensions (rows x cols): (" << tensor_eigen.dimension(0)
              << "x" << tensor_eigen.dimension(1) << ")" << std::endl;

    // Perform the contraction operation: `np.einsum("abc, bd -> acd", update, self.tensor)`
    Eigen::array<Eigen::IndexPair<int>, 1> contraction_dims = {Eigen::IndexPair<int>(1, 0)};
    Eigen::Tensor<std::complex<double>, 3> result_tensor = update_tensor.contract(tensor_eigen, contraction_dims);

    // Print result_tensor dimensions after contraction
    std::cout << "Result tensor dimensions after contraction: ("
              << result_tensor.dimension(0) << "x"
              << result_tensor.dimension(1) << "x"
              << result_tensor.dimension(2) << ")" << std::endl;

    // Reshape result_tensor to 2D tensor
    Eigen::Tensor<std::complex<double>, 2> reshaped_tensor = result_tensor.reshape(
        Eigen::array<Eigen::Index, 2>{physical_dim, gate_dim * tensor_eigen.dimension(1)});

    std::cout << "Reshaped tensor dimensions: (" << reshaped_tensor.dimension(0) << "x"
              << reshaped_tensor.dimension(1) << ")" << std::endl;

    tensor_matrix_ = tensorToMatrix(reshaped_tensor);
    std::cout << "Final reshaped tensor dimensions: (" << tensor_matrix_->rows() << "x"
              << tensor_matrix_->cols() << ")" << std::endl;
}

std::shared_ptr<TNode> TNode::getItem(const std::string& key) {

    // Start from the root and use BFS to find the node with the given key
    std::queue<std::shared_ptr<TNode>> node_queue;
    node_queue.push(findRoot());

    while (!node_queue.empty()) {
        auto current_node = node_queue.front();
        node_queue.pop();

        if (current_node->getName() == key) {
            return current_node;
        }

        for (const auto& child : current_node->getChildren()) {
            node_queue.push(child);
        }
    }
    std::cerr << "Node with key " << key << " not found." << std::endl;
    return nullptr;
}

std::vector<std::shared_ptr<TNode>> TNode::getItem(const std::string& site_i, const std::string& site_j) {
    std::queue<std::shared_ptr<TNode>> node_queue;
    node_queue.push(findRoot());

    std::unordered_map<std::shared_ptr<TNode>, std::shared_ptr<TNode>> parent_map;
    std::shared_ptr<TNode> node_i = nullptr;
    std::shared_ptr<TNode> node_j = nullptr;

    // Perform a BFS to locate site_i and site_j and build the parent map
    while (!node_queue.empty()) {
        auto current_node = node_queue.front();
        node_queue.pop();

        if (current_node->getName() == site_i) {
            node_i = current_node;
        }
        if (current_node->getName() == site_j) {
            node_j = current_node;
        }

        if (node_i && node_j) break;

        for (const auto& child : current_node->getChildren()) {
            node_queue.push(child);
            parent_map[child] = current_node;
        }
    }

    if (!node_i || !node_j) {
        std::cerr << "Error: Sites " << site_i << " and/or " << site_j << " not found." << std::endl;
        return {};
    }

    // Build paths from site_i and site_j to the root
    std::vector<std::shared_ptr<TNode>> path_i, path_j;

    auto temp = node_i;
    while (temp) {
        path_i.push_back(temp);
        temp = parent_map[temp];
    }

    temp = node_j;
    while (temp) {
        path_j.push_back(temp);
        temp = parent_map[temp];
    }

    // Reverse the paths to go from root to the nodes
    std::reverse(path_i.begin(), path_i.end());
    std::reverse(path_j.begin(), path_j.end());

    // Find the common ancestor and merge paths
    int idx_i = 0, idx_j = 0;
    while (idx_i < path_i.size() && idx_j < path_j.size() && path_i[idx_i] == path_j[idx_j]) {
        ++idx_i;
        ++idx_j;
    }

    // Construct the full path from site_i to site_j via the common ancestor
    std::vector<std::shared_ptr<TNode>> full_path;
    full_path.insert(full_path.end(), path_i.begin() + idx_i, path_i.end());
    full_path.insert(full_path.end(), path_j.rbegin(), path_j.rend() - idx_j);

    // Debugging: Output the full path
    std::cout << "Complete path from site " << site_i << " to site " << site_j << ": ";
    for (const auto& node : full_path) {
        std::cout << node->getName() << " ";
    }
    std::cout << std::endl;

    return full_path;
}

std::vector<std::shared_ptr<TNode>> TNode::getPathToRoot(const std::string& site) {
    std::vector<std::shared_ptr<TNode>> path;
    auto current_node = getItem(site);

    while (current_node) {
        path.push_back(current_node);
        current_node = current_node->getParent();
    }

    return path;
}

std::vector<std::shared_ptr<TNode>> TNode::getIntermediateNodes(const std::string& site_i, const std::string& site_j) {
    // Get paths from both nodes up to the root
    auto path_i = getPathToRoot(site_i);
    auto path_j = getPathToRoot(site_j);

    // Determine where the paths start diverging
    int idx_i = path_i.size() - 1;
    int idx_j = path_j.size() - 1;

    // Find the first common ancestor by moving from the root until paths diverge
    std::shared_ptr<TNode> common_ancestor = nullptr;
    while (idx_i >= 0 && idx_j >= 0 && path_i[idx_i] == path_j[idx_j]) {
        common_ancestor = path_i[idx_i];
        idx_i--;
        idx_j--;
    }

    // Collect intermediate nodes by building the path from site_i to site_j
    std::set<std::shared_ptr<TNode>> unique_intermediate_nodes;  // Use a set to avoid duplicates

    // From site_i to the first common ancestor, excluding leaf nodes
    for (int i = 0; i <= idx_i + 1; ++i) {
        if (!path_i[i]->isLeaf()) {
            unique_intermediate_nodes.insert(path_i[i]);
        }
    }

    // Add the common ancestor if it's not a leaf
    if (common_ancestor && !common_ancestor->isLeaf()) {
        unique_intermediate_nodes.insert(common_ancestor);
    }

    // From the common ancestor to site_j, excluding leaf nodes
    for (int j = idx_j; j >= 0; --j) {
        if (!path_j[j]->isLeaf()) {
            unique_intermediate_nodes.insert(path_j[j]);
        }
    }

    // Convert the set to a vector to preserve the order in which nodes were added
    std::vector<std::shared_ptr<TNode>> intermediate_nodes(unique_intermediate_nodes.begin(), unique_intermediate_nodes.end());

    // Debug output
    std::cout << "Intermediate nodes between sites " << site_i << " and " << site_j << ": ";
    for (const auto& node : intermediate_nodes) {
        std::cout << node->getName() << " ";
    }
    std::cout << std::endl;

    return intermediate_nodes;
}

void TNode::update(int gate_dim, int site_i, int site_j) {
    assert(!isLeaf());

    if (isRoot()) {
        tmp_index_ = site_i;
        tmp_dim_ = gate_dim;
        std::cout << "Root node update: setting tmp_index to " << site_i
                  << " and tmp_dim to " << gate_dim << std::endl;
        return;
    }

    // Identify indices for expansion
    int index_i = leaf_indices_.count(std::to_string(site_i))
                  ? leaf_indices_.at(std::to_string(site_i))
                  : tensor_tensor_->dimension(0) - 1;
    int index_j = leaf_indices_.count(std::to_string(site_j))
                  ? leaf_indices_.at(std::to_string(site_j))
                  : tensor_tensor_->dimension(2) - 1;

    std::cout << "Indices for expansion: [" << index_i << ", " << index_j << "]" << std::endl;

    // Expand and contract dynamically
    for (int idx : {index_i, index_j}) {
        // Get current dimensions
        Eigen::array<Eigen::Index, 3> current_dims = {
            tensor_tensor_->dimension(0),
            tensor_tensor_->dimension(1),
            tensor_tensor_->dimension(2)};
        std::cout << "Current tensor dimensions before expansion: ["
                  << current_dims[0] << "x" << current_dims[1] << "x" << current_dims[2] << "]"
                  << std::endl;

        // Expand the specified dimension
        Eigen::array<Eigen::Index, 3> expanded_dims = current_dims;
        expanded_dims[idx] *= gate_dim;

        Eigen::Tensor<std::complex<double>, 3> expanded_tensor(expanded_dims);
        expanded_tensor.setZero();

        for (int i = 0; i < current_dims[0]; ++i) {
            for (int j = 0; j < current_dims[1]; ++j) {
                for (int k = 0; k < current_dims[2]; ++k) {
                    for (int g = 0; g < gate_dim; ++g) {
                        if (idx == 0) {
                            expanded_tensor(i * gate_dim + g, j, k) = tensor_tensor_->operator()(i, j, k);
                        } else if (idx == 1) {
                            expanded_tensor(i, j * gate_dim + g, k) = tensor_tensor_->operator()(i, j, k);
                        } else if (idx == 2) {
                            expanded_tensor(i, j, k * gate_dim + g) = tensor_tensor_->operator()(i, j, k);
                        }
                    }
                }
            }
        }

        std::cout << "Tensor dimensions after expansion: [" << expanded_dims[0] << "x"
                  << expanded_dims[1] << "x" << expanded_dims[2] << "]" << std::endl;

        // Create identity tensor for contraction
        Eigen::Tensor<std::complex<double>, 2> identity(gate_dim, gate_dim);
        identity.setZero();
        for (int i = 0; i < gate_dim; ++i) {
            identity(i, i) = std::complex<double>(1.0, 0.0);  // Set diagonal to 1
        }

        // Perform contraction
        Eigen::array<Eigen::IndexPair<int>, 1> contraction_pair = {Eigen::IndexPair<int>(idx, 0)};
        Eigen::Tensor<std::complex<double>, 3> contracted_tensor =
            expanded_tensor.contract(identity, contraction_pair);

        tensor_tensor_ = contracted_tensor;

        std::cout << "Tensor dimensions after contraction along index " << idx << ": ["
                  << tensor_tensor_->dimension(0) << "x" << tensor_tensor_->dimension(1) << "x"
                  << tensor_tensor_->dimension(2) << "]" << std::endl;
    }

    // Final tensor dimensions
    std::cout << "Final tensor dimensions after update: [" << tensor_tensor_->dimension(0)
              << "x" << tensor_tensor_->dimension(1) << "x" << tensor_tensor_->dimension(2)
              << "]" << std::endl;
}

std::vector<int> TNode::getTensorDimensions() const {
    if (tensor_matrix_) {
        return {static_cast<int>(tensor_matrix_->rows()), static_cast<int>(tensor_matrix_->cols())};
    }
     if (tensor_tensor_) {
        auto dims = tensor_tensor_->dimensions();
        return std::vector<int>(dims.data(), dims.data() + dims.size());
    }

    return {};
}
