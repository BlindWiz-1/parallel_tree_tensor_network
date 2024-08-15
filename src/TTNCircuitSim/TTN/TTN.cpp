#include "TTN.h"
#include "../TNode/TNode.h"
#include "../../Structure/SNode/SNode.h"
#include "../../Structure/FindTreeStructure/FindTreeStructure.h"
#include "../../Circuits/Circuit/Circuit.h"
#include "../../Circuits/SingleStateToTrees/SingleStateToTrees.h"
#include "../../Operations/TTNContract/TTNContract.h"
#include "../../Operations/Orthonormalization/Orthonormalization.h"
#include "../TreeStructure/TreeStructure.h"
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>

TTN::TTN(int local_dim, std::shared_ptr<TNode> root, int d_max, bool enable_gpu, bool dry_run)
    : local_dimension_(local_dim), root_(root), nrm_(1.0), d_max_(d_max), enable_gpu_(enable_gpu), dry_run_(dry_run) {}

int TTN::localDim() const {
    return local_dimension_;
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
    return root_->getTensor().cast<double>();
}

std::shared_ptr<TTN> TTN::basisState(int local_dim,
    const std::vector<int>& single_states,
    const std::shared_ptr<SNode>& structure,
    const std::optional<Circuit>& circ,
    int d_max,
    int clusters,
    bool flat,
    int bound
    ) {

    assert(structure != nullptr || circ.has_value());

    std::shared_ptr<SNode> final_structure = structure;
    if (structure == nullptr) {

        std::unordered_map<std::shared_ptr<SNode>, TreeStructure> structures;
        for (int i = 2; i < std::min(static_cast<int>(circ->getLSites() / 2), 15); ++i) {
            auto tmp_structure = findTreeStructure(*circ, i, single_states.size(), bound, flat);
            auto root = createPseudoTree(tmp_structure, single_states, local_dim);
            std::shared_ptr<TTN> psi = std::make_shared<TTN>(local_dim, root, d_max, false, true);
            applyCircuit(*psi, *circ);
            auto [tmp_cumulative, max_bond] = psi->bondData();
            structures[tmp_structure] = TreeStructure(i, max_bond, psi->maxLeaves(), tmp_cumulative);
        }
        final_structure = findBestStructure(structures, arguments);
    }

    if (arguments.count("dry_run") && arguments.at("dry_run")) {
        auto root = createPseudoTree(final_structure, single_states, local_dim);
        return std::make_shared<TTN>(local_dim, root, d_max, false, true);
    }

    auto root = singleStatesToTree(single_states, local_dim, final_structure);
    return std::make_shared<TTN>(local_dim, root, d_max, arguments.count("enable_gpu") && arguments.at("enable_gpu"));
}

Eigen::VectorXd TTN::asVector() const {
    if (dry_run_) {
        Eigen::VectorXd vec(nSites());
        vec.setZero();
        vec(nSites() - 1) = 1;
        return vec;
    }
    Eigen::MatrixXd state = contract(root_, nrm_, enable_gpu_);
    auto axes = root_->getLeafIndices();
    std::vector<int> axes_values;
    for (const auto& kv : axes) {
        axes_values.push_back(kv.second);
    }
    std::sort(axes_values.begin(), axes_values.end());
    // Reshape and reorder the state vector according to the axes
    return state;
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
    auto local_orthonormalize = dry_run_ ? pseudoOrthonormalize : (compress ? orthonormalizeSVD : orthonormalizeQR);

    Eigen::MatrixXd factor;
    for (auto node = root_->getItem(site_i); node != nullptr; node = node->getParent()) {
        if (node->getLeafIndices().count(std::to_string(site_j))) {
            if (dry_run_) {
                node->updateShape(site_i, factor);
            } else if (node->isRoot()) {
                node->setTmpFactor(factor);
            } else {
                node = contractFactorOnIndex(node, factor, site_i);
            }
            break;
        }
        factor = local_orthonormalize(node, site_i, factor);
        if (isSquareIdentity(factor)) break;
    }

    factor = Eigen::MatrixXd();
    for (auto node = root_->getItem(site_j); node != nullptr; node = node->getParent()) {
        if (node->isRoot() && !node->getTmpFactor().empty()) {
           precontractRoot(node, site_j, factor);
            factor = Eigen::MatrixXd();
        }
        factor = local_orthonormalize(node, site_j, factor);
        if (isSquareIdentity(factor)) break;
    }

    if (!dry_run_ && factor.size() > 0) {
        nrm_ *= factor(0, 0);
    }
}

bool TTN::isSquareIdentity(const Eigen::MatrixXd& factor) const {
    if (factor.size() == 0 || factor.rows() != factor.cols()) {
        return false;
    }
    return factor.isApprox(Eigen::MatrixXd::Identity(factor.rows(), factor.cols()));
}

void TTN::applyCircuit(const std::vector<std::shared_ptr<TNode>>& circuit) {
    for (const auto& gate : circuit) {
        std::cout << "Applying gate with shape: " << gate->getTensor().rows() << "x" << gate->getTensor().cols() << std::endl;
        root_->applyGate(gate->getTensor());
    }
}

void TTN::display() const {
    root_->display();
}
