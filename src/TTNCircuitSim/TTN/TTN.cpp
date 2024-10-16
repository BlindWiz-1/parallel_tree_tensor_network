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

Eigen::MatrixXd TTN::dtype() const {
    return root_->getTensor().real();
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
    Eigen::Matrix<std::complex<double>, -1, -1> factor;  // Stores the factor after normalization
    std::optional<Eigen::Matrix<std::complex<double>, -1, -1>> tmp_factor;

    std::cout << "Starting orthonormalization from site " << site_i << " to site " << site_j << std::endl;

    // **First Loop: Traverse from site_i upwards**
    for (auto node = root_->getItem(site_i); node != nullptr; node = node->getParent()) {
        std::cout << "Node dimensions at site " << node->getName() << " before normalization: (" << node->getTensor().rows() << "x" << node->getTensor().cols() << ")" << std::endl;

        // Check if site_j is in the current node's leaf indices (indicating common ancestor)
        if (node->getLeafIndices().count(std::to_string(site_j))) {
            // If the current node is the root, store the factor in tmp_factor
            if (node->isRoot()) {
                node->setTmpFactor(factor);
            } else {
                // For non-root nodes, contract the factor into the node's tensor
                contractFactorOnIndex(node->getTensor(), factor, site_i);
            }
            break;  // Early exit since we've found the common ancestor
        }

        // Perform orthonormalization and update the factor
        factor = orthonormalizeSVD(*node, site_i, d_max_);
        std::cout << "Factor dimensions after normalization: (" << factor.rows() << "x" << factor.cols() << ")" << std::endl;

        // Stop if the factor is a square identity matrix (early stopping)
        if (isSquareIdentity(factor)) {
            std::cout << "Early stopping: Factor is a square identity matrix." << std::endl;
            break;
        }
    }

    // Reset the factor for the second loop
    factor = Eigen::Matrix<std::complex<double>, -1, -1>();

    // **Second Loop: Traverse from site_j upwards**
    for (auto node = root_->getItem(site_j); node != nullptr; node = node->getParent()) {
        std::cout << "Node dimensions at start of site j " << node->getName() << " before normalization: (" << node->getTensor().rows() << "x" << node->getTensor().cols() << ")" << std::endl;

        // Special handling for the root node during site_j traversal
        if (node->isRoot() && node->getTmpFactor()->rows() != 0 && node->getTmpFactor()->cols() != 0) {
            std::cout << "Handling root contraction." << std::endl;
            precontractRoot(*node, site_j, factor);  // Contract the stored factor at the root
            factor = Eigen::MatrixXcd();  // Reset the factor again after root contraction
        }

        // Perform orthonormalization and update the factor for site_j
        factor = orthonormalizeSVD(*node, site_j, d_max_);
        std::cout << "Factor dimensions after normalization: (" << factor.rows() << "x" << factor.cols() << ")" << std::endl;

        // Stop if the factor is a square identity matrix (early stopping)
        if (isSquareIdentity(factor)) {
            std::cout << "Early stopping: Factor is a square identity matrix." << std::endl;
            break;
        }
    }

    // If the final factor is valid, update the normalization factor
    if (factor.size() > 0) {
        nrm_ *= factor(0, 0).real();
        std::cout << "Updated normalization factor: " << nrm_ << std::endl;
    }
}

bool TTN::isSquareIdentity(const Tensor& factor) const {
    if (factor.size() == 0 || factor.rows() != factor.cols()) {
        return false;
    }
    return factor.isApprox(Eigen::MatrixXd::Identity(factor.rows(), factor.cols()));
}

void TTN::display() const {
    root_->display();
}
