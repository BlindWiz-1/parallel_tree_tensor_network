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

    // Reshape tensor into the desired shape: (local_dim, flattened_dimension)
    Eigen::array<Eigen::Index, 2> shape = {local_dimension_, static_cast<int>(axes_values.size())};
    Eigen::Tensor<std::complex<double>, 2> reshaped_tensor = tensor_state.reshape(shape);

    // Apply permutation if needed (you may need to adjust the permutation indices)
    Eigen::array<int, 2> permute_indices = {0, 1};  // This should be adjusted to match your specific reordering logic
    Eigen::Tensor<std::complex<double>, 2> permuted_tensor = reshaped_tensor.shuffle(permute_indices).eval(); // Use eval() to create a concrete tensor

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
    Eigen::Matrix<std::complex<double>, -1, -1> factor;
    std::optional<Eigen::Matrix<std::complex<double>, -1, -1>> tmp_factor;

    std::cout << "Starting orthonormalization from site " << site_i << " to site " << site_j << std::endl;

    // Process site_i upwards
    for (auto node = root_->getItem(site_i); node != nullptr; node = node->getParent()) {
        std::cout << "Node dimensions before normalization: (" << node->getTensor().rows() << "x" << node->getTensor().cols() << ")" << std::endl;
        if (node->getLeafIndices().count(std::to_string(site_j))) {
            if (node->isRoot()) {
                node->setTmpFactor(factor);
            } else {
                contractFactorOnIndex(node->getTensor(), factor, site_i);
            }
            break;
        }
        factor = orthonormalizeSVD(node->getTensor(), site_i, d_max_, node->getTmpFactor(), node->getLeafIndices());
        std::cout << "Factor dimensions after normalization: (" << factor.rows() << "x" << factor.cols() << ")" << std::endl;
        if (isSquareIdentity(factor)) break;
    }

    // Process site_j upwards
    factor = Eigen::Matrix<std::complex<double>, -1, -1>();  // Reset factor
    for (auto node = root_->getItem(site_j); node != nullptr; node = node->getParent()) {
        std::cout << "Processing node at site " << site_j << "..." << std::endl;
        if (node->isRoot() && node->getTmpFactor()->rows() == 0 && node->getTmpFactor()->cols() == 0) {
            std::cout << "Precontracting root..." << std::endl;
            precontractRoot(*node, site_j, factor);
            factor = Eigen::MatrixXcd();  // Reset factor again
        }

        factor = orthonormalizeSVD(node->getTensor(), site_j, d_max_, node->getTmpFactor(), node->getLeafIndices());
        if (isSquareIdentity(factor)) break;
    }

    // Update normalization factor if necessary
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
