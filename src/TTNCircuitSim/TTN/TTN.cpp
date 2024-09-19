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

    auto root = singleStatesToTree(single_states, local_dim, findTreeStructure(*circ, single_states.size(), flat));
    auto ttn = std::make_shared<TTN>(local_dim, root, d_max);
    ttn->display();
    return ttn;
}

Eigen::VectorXd TTN::asVector() const {
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> state = contract(root_, nrm_);
    auto axes = root_->getLeafIndices();
    std::vector<int> axes_values;
    for (const auto& kv : axes) {
        axes_values.push_back(kv.second);
    }
    std::sort(axes_values.begin(), axes_values.end());
    // Reshape and reorder the state vector according to the axes
    return state.real();
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

void TTN::orthonormalize(int site_i, int site_j, bool compress, double tol) {
    Eigen::Matrix<std::complex<double>, -1, -1> factor;
    std::optional<Eigen::Matrix<std::complex<double>, -1, -1>> tmp_factor;

    // Process site_i
    for (auto node = root_->getItem(site_i); node != nullptr; node = node->getParent()) {
        if (node->getLeafIndices().count(std::to_string(site_j))) {
             if (node->isRoot()) {
                node->setTmpFactor(factor);
            } else {
                contractFactorOnIndex(node->getTensor(), factor, site_i);
            }
            break;
        }

        factor = orthonormalizeSVD(node->getTensor(), site_i, d_max_, node->getTmpFactor(), node->getLeafIndices());
        if (isSquareIdentity(factor)) break;
    }

    // Process site_j
    factor = Eigen::Matrix<std::complex<double>, -1, -1>();
    for (auto node = root_->getItem(site_j); node != nullptr; node = node->getParent()) {
        if (node->isRoot() && node->getTmpFactor()->rows() == 0 && node->getTmpFactor()->cols() == 0) {
            precontractRoot(*node, site_j, factor);
            factor = Eigen::MatrixXd();
        }

        factor = orthonormalizeSVD(node->getTensor(), site_i, d_max_, node->getTmpFactor(), node->getLeafIndices());
        if (isSquareIdentity(factor)) break;
    }

    // Update normalization factor
    if (factor.size() > 0) {
        nrm_ *= factor(0, 0).real();
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
