#include "TTNContract.h"
#include "../../TTNCircuitSim/TNode/TNode.h"

Tensor contract(const std::shared_ptr<TNode>& node, double nrm, bool enable_gpu) {
    if (node->isLeaf()) {
        // Leaves are already contracted
        return node->getTensor();
    }

    // Add current parent tensor and label children [0,...,ndim]
    int counter = node->getTensor().cols();
    auto params = {node->getTensor()};
    std::vector<Eigen::Index> indices;

    for (int i = 0; i < node->getTensor().cols(); ++i) {
        indices.push_back(i);
    }

    for (size_t idx = 0; idx < node->getChildren().size(); ++idx) {
        // Add each contracted child tensor
        auto child = node->getChildren()[idx];
        Tensor child_tensor = contract(child, 1, enable_gpu);

        // Place its already contracted leaves correctly
        std::vector<Eigen::Index> child_indices;
        for (int i = 0; i < child_tensor.cols() - 1; ++i) {
            child_indices.push_back(counter++);
        }
        child_indices.push_back(idx);

        // Append child tensor and its indices
        params.push_back(child_tensor);
        for (const auto& ci : child_indices) {
            indices.push_back(ci);
        }
    }

    if (!node->isRoot()) {
        // Keep parent index
        indices.back() = counter;
    } else {
        // Or add normalization factor
        params.push_back(Tensor::Constant(1, 1, nrm));
        indices.push_back(node->getTensor().cols() - 1);
    }

    std::cout << "Contracting node " << node->getName() << std::endl;

    Tensor result;
    if (enable_gpu) {
        result = Tensor::Constant(1, 1, nrm);
    } else {
        result = params[0];
        for (size_t i = 1; i < params.size(); ++i) {
            result = result * params[i];
        }
    }
    return result;
}
