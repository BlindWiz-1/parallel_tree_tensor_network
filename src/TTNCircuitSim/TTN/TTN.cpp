#include "TTN.h"
#include "../TNode/TNode.h"
#include "../../Structure/SNode/SNode.h"
#include "../../Structure/FindTreeStructure/FindTreeStructure.h"
#include "../../Circuits/Circuit/Circuit.h"
#include "../../Circuits/SingleStateToTrees/SingleStateToTrees.h"
#include "../../Operations/TTNContract/TTNContract.h"
#include "../../Operations/Orthonormalization/Orthonormalization.h"
#include "../TreeStructure/TreeStructure.h"
#include <utility>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

TTN::TTN(int local_dim, std::shared_ptr<TNode> root, int d_max)
    : local_dimension_(local_dim), root_(std::move(root)), nrm_(1.0), d_max_(d_max) {}

int TTN::localDim() const {
    return local_dimension_;
}

std::shared_ptr<TNode> TTN::getTNodeRoot() const {
    return root_;
}

void TTN::setTNodeRoot(const std::shared_ptr<TNode>& root) {
    root_ = root;
}

int TTN::nSites() const {
    // Recursive function to count the number of leaf nodes
    std::function<int(const std::shared_ptr<TNode>&)> countLeaves = [&](const std::shared_ptr<TNode>& node) -> int {
        if (node->isLeaf()) return 1;
        int count = 0;
        for (const auto& child : node->getChildren()) {
            count += countLeaves(child);
        }
        return count;
    };
    return countLeaves(root_);
}

std::shared_ptr<TTN> TTN::basisState(int local_dim,
    const std::vector<int>& single_states,
    const std::shared_ptr<SNode>& structure,
    const std::optional<Circuit>& circ,
    int d_max, bool flat
    ) {

    assert(structure != nullptr || circ.has_value());

    // If no structure is provided, build it based on the circuit
    auto actual_structure = structure;
    if (!structure) {
        actual_structure = findTreeStructure(*circ, single_states.size(), flat);
    }

    auto root = singleStatesToTree(single_states, local_dim, actual_structure);
    auto ttn = std::make_shared<TTN>(local_dim, root, d_max);
    ttn->display();
    return ttn;
}

Eigen::VectorXd TTN::asVector() const {
    // Contract the tree tensor network and get the resulting state
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> state = contract(root_, nrm_);

    std::cout << "Initial contracted state dimensions: (" << state.rows() << "x" << state.cols() << ")" << std::endl;

    // Extract and sort the axes
    auto axes = root_->getLeafIndices();
    std::vector<int> axes_values;
    for (const auto& kv : axes) {
        axes_values.push_back(kv.second);
    }
    std::sort(axes_values.begin(), axes_values.end());

    // Convert Eigen::Matrix to Eigen::Tensor
    Eigen::Tensor<std::complex<double>, 2> tensor_state(state.rows(), state.cols());
    for (Eigen::Index i = 0; i < state.rows(); ++i) {
        for (Eigen::Index j = 0; j < state.cols(); ++j) {
            tensor_state(i, j) = state(i, j);
        }
    }

    std::cout << "Tensor state dimensions before reshape: (" << tensor_state.dimension(0) << "x" << tensor_state.dimension(1) << ")" << std::endl;

    // Calculate the target shape based on local dimensions and number of leaves
    Eigen::array<Eigen::Index, 2> shape = {local_dimension_, static_cast<int>(axes_values.size())};
    std::cout << "Target reshape dimensions: (" << shape[0] << "x" << shape[1] << ")" << std::endl;

    // Check that the number of elements match before reshaping
    assert(tensor_state.size() == shape[0] * shape[1] && "Mismatch in element count for reshape.");

    // Reshape tensor into the desired shape: (local_dim, flattened_dimension)
    Eigen::Tensor<std::complex<double>, 2> reshaped_tensor = tensor_state.reshape(shape);

    // Debug reshape result
    std::cout << "Reshaped tensor dimensions: (" << reshaped_tensor.dimension(0) << "x" << reshaped_tensor.dimension(1) << ")" << std::endl;

    // Apply permutation if needed (adjust permutation indices accordingly)
    Eigen::array<int, 2> permute_indices = {0, 1};  // Modify as necessary
    Eigen::Tensor<std::complex<double>, 2> permuted_tensor = reshaped_tensor.shuffle(permute_indices).eval(); // Use eval() to create a concrete tensor

    // Check the size after shuffling
    std::cout << "Permuted tensor dimensions: (" << permuted_tensor.dimension(0) << "x" << permuted_tensor.dimension(1) << ")" << std::endl;

    // Convert Eigen::Tensor back to Eigen::VectorXd for output
    Eigen::VectorXd final_vector(permuted_tensor.size());
    for (Eigen::Index i = 0; i < permuted_tensor.size(); ++i) {
        final_vector(i) = permuted_tensor(i).real();
    }

    return final_vector;
}

std::pair<float, int> TTN::bondData() const {
    return root_->countDimensions();
}

int TTN::maxLeaves() const {
    std::function<int(const std::shared_ptr<TNode>&)> countMaxLeaves = [&](const std::shared_ptr<TNode>& node) -> int {
        if (node->isLeaf()) return 1;
        int max_leaves = 0;
        for (const auto& child : node->getChildren()) {
            max_leaves = std::max(max_leaves, countMaxLeaves(child));
        }
        return max_leaves;
    };
    return countMaxLeaves(root_);
}

void TTN::orthonormalize(int site_i, int site_j) {
    Eigen::MatrixXcd factor;  // Stores the factor after normalization

    std::cout << "Starting orthonormalization from site " << site_i << " to site " << site_j << std::endl;

    // **First Loop: Traverse from site_i upwards**
    for (auto node = root_->getItem(std::to_string(site_i)); node != nullptr; node = node->getParent()) {
        auto tensor_variant = node->getTensor();

        // Print tensor dimensions based on the type
        if (std::holds_alternative<Eigen::MatrixXcd>(tensor_variant)) {
            const auto& tensor_matrix = std::get<Eigen::MatrixXcd>(tensor_variant);
            std::cout << "Node dimensions at site " << node->getName() << " before normalization: ("
                      << tensor_matrix.rows() << "x" << tensor_matrix.cols() << ")" << std::endl;
        } else if (std::holds_alternative<Eigen::Tensor<std::complex<double>, 3>>(tensor_variant)) {
            const auto& tensor_tensor = std::get<Eigen::Tensor<std::complex<double>, 3>>(tensor_variant);
            std::cout << "Node dimensions at site " << node->getName() << " before normalization: ("
                      << tensor_tensor.dimension(0) << "x" << tensor_tensor.dimension(1) << "x" << tensor_tensor.dimension(2) << ")" << std::endl;
        }

        // Stop when the common ancestor with `site_j` is found
        if (node->getLeafIndices().count(std::to_string(site_j))) {
            if (node->isRoot()) {
                node->setTmpFactor(factor);
            } else {
                contractFactorOnIndex(node, factor, site_i);
            }
            break;
        }

        // Perform SVD-based orthonormalization and update the factor
        factor = orthonormalizeSVD(node, site_i, d_max_);
        std::cout << "Factor dimensions after normalization: (" << factor.rows() << "x" << factor.cols() << ")" << std::endl;

        if (isSquareIdentity(factor)) {
            std::cout << "Early stopping: Factor is a square identity matrix." << std::endl;
            break;
        }
    }

    // Reset the factor for the second loop
    factor = Eigen::MatrixXcd();

    // **Second Loop: Traverse from site_j upwards**
    for (auto node = root_->getItem(std::to_string(site_j)); node != nullptr; node = node->getParent()) {
        auto tensor_variant = node->getTensor();

        if (std::holds_alternative<Eigen::MatrixXcd>(tensor_variant)) {
            const auto& tensor_matrix = std::get<Eigen::MatrixXcd>(tensor_variant);
            std::cout << "Node dimensions at start of site j " << node->getName() << " before normalization: ("
                      << tensor_matrix.rows() << "x" << tensor_matrix.cols() << ")" << std::endl;
        } else if (std::holds_alternative<Eigen::Tensor<std::complex<double>, 3>>(tensor_variant)) {
            const auto& tensor_tensor = std::get<Eigen::Tensor<std::complex<double>, 3>>(tensor_variant);
            std::cout << "Node dimensions at start of site j " << node->getName() << " before normalization: ("
                      << tensor_tensor.dimension(0) << "x" << tensor_tensor.dimension(1) << "x" << tensor_tensor.dimension(2) << ")" << std::endl;
        }

        // Handle special case for root node during traversal from site_j
        if (node->isRoot() && node->getTmpFactor().has_value() && node->getTmpFactor()->rows() != 0 && node->getTmpFactor()->cols() != 0) {
            std::cout << "Handling root contraction." << std::endl;
            precontractRoot(node, site_j, factor);
            factor = Eigen::MatrixXcd();
        }

        // Perform SVD-based orthonormalization and update the factor for site_j
        factor = orthonormalizeSVD(node, site_j, d_max_);
        std::cout << "Factor dimensions after normalization: (" << factor.rows() << "x" << factor.cols() << ")" << std::endl;

        if (isSquareIdentity(factor)) {
            std::cout << "Early stopping: Factor is a square identity matrix." << std::endl;
            break;
        }
    }

    if (factor.size() > 0) {
        nrm_ *= factor(0, 0).real();
        std::cout << "Updated normalization factor: " << nrm_ << std::endl;
    }
}

bool TTN::isSquareIdentity(const Eigen::MatrixXcd& factor) const {
    if (factor.size() == 0 || factor.rows() != factor.cols()) {
        return false;
    }
    return factor.isApprox(Eigen::MatrixXcd::Identity(factor.rows(), factor.cols()));
}

void TTN::display() const {
    root_->display();
}
