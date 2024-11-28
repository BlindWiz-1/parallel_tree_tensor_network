#include "TTNContract.h"
#include "../../TTNCircuitSim/TNode/TNode.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

// Helper function: Converts Eigen::MatrixXcd to Eigen::Tensor
Eigen::Tensor<std::complex<double>, 2> toTensor(const Eigen::MatrixXcd& matrix) {
    return Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 2>>(const_cast<std::complex<double>*>(matrix.data()), matrix.rows(), matrix.cols());
}

// Helper function: Converts Eigen::Tensor to Eigen::MatrixXcd
Eigen::MatrixXcd toMatrix(const Eigen::Tensor<std::complex<double>, 2>& tensor) {
    return Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>>(tensor.data(), tensor.dimension(0), tensor.dimension(1));
}

Tensor contract(const std::shared_ptr<TNode>& node, double nrm) {

    if (node->isLeaf()) {
        std::cout << "Leaf node detected: " << node->getName() << std::endl;
        auto tensor_variant = node->getTensor();

        if (std::holds_alternative<Eigen::MatrixXcd>(tensor_variant)) {
            return std::get<Eigen::MatrixXcd>(tensor_variant);  // Leaves are already contracted as Matrix
        }
    }

    std::vector<Eigen::Tensor<std::complex<double>, 2>> child_tensors;

    // Recursively contract all child nodes
    for (const auto& child : node->getChildren()) {
        auto child_tensor = toTensor(contract(child, 1));  // Recursively contract child
        child_tensors.push_back(child_tensor);
    }

    auto parent_tensor_variant = node->getTensor();
    Eigen::Tensor<std::complex<double>, 2> parent_tensor;
    parent_tensor = std::get<Eigen::Tensor<std::complex<double>, 3>>(parent_tensor_variant).chip(0, 2);

    std::cout << "Parent tensor dimensions before contraction: ("
              << parent_tensor.dimension(0) << "x" << parent_tensor.dimension(1) <<  "x" << parent_tensor.dimension(2)  <<  ")" << std::endl;

    for (const auto& child_tensor : child_tensors) {
        std::cout << "Contracting with child tensor dimensions: ("
                  << child_tensor.dimension(0) << "x" << child_tensor.dimension(1) << ")" << std::endl;

        // Dynamically set contraction dimensions
        Eigen::array<Eigen::IndexPair<int>, 1> contraction_dims = {Eigen::IndexPair<int>(1, 0)};  // Adjust based on tensor structure

        // Check if dimensions match for contraction
        if (parent_tensor.dimension(1) != child_tensor.dimension(0)) {
            std::cerr << "Dimension mismatch detected: parent_tensor dimension "
                      << parent_tensor.dimension(1) << " and child tensor dimension "
                      << child_tensor.dimension(0) << std::endl;
            continue;  // Skip contracting this tensor if there's a mismatch
        }

        // Perform the contraction
        parent_tensor = parent_tensor.contract(child_tensor, contraction_dims);

        std::cout << "Result tensor dimensions after contraction: ("
                  << parent_tensor.dimension(0) << "x" << parent_tensor.dimension(1) << ")" << std::endl;
    }

    // Apply normalization if this is the root node
    if (node->isRoot()) {
        Eigen::Tensor<std::complex<double>, 2> norm_tensor(1, 1);
        norm_tensor(0, 0) = std::complex<double>(nrm, 0.0);
        parent_tensor = parent_tensor.contract(norm_tensor, Eigen::array<Eigen::IndexPair<int>, 1>{});
    }

    Eigen::MatrixXcd final_matrix = toMatrix(parent_tensor);
    return final_matrix;
}
