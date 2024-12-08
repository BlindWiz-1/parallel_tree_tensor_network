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

Eigen::Tensor<std::complex<double>, 2> matrixToTensor(
    const Eigen::MatrixXcd& matrix,
    Eigen::Index rows,
    Eigen::Index cols) {
    Eigen::Tensor<std::complex<double>, 2> tensor(rows, cols);

    for (Eigen::Index i = 0; i < rows; ++i) {
        for (Eigen::Index j = 0; j < cols; ++j) {
            tensor(i, j) = matrix(i, j);
        }
    }

    return tensor;
}

Eigen::MatrixXcd contract(const std::shared_ptr<TNode>& node, double nrm) {
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

    if (std::holds_alternative<Eigen::MatrixXcd>(parent_tensor_variant)) {
        auto matrix = std::get<Eigen::MatrixXcd>(parent_tensor_variant);
        parent_tensor = matrixToTensor(matrix, matrix.rows(), matrix.cols());
    } else if (std::holds_alternative<Eigen::Tensor<std::complex<double>, 3>>(parent_tensor_variant)) {
        parent_tensor = std::get<Eigen::Tensor<std::complex<double>, 3>>(parent_tensor_variant).chip(0, 2);
    } else {
        throw std::runtime_error("Unsupported tensor format in contract");
    }

    std::cout << "Parent tensor dimensions before contraction: ("
              << parent_tensor.dimension(0) << "x" << parent_tensor.dimension(1) << ")" << std::endl;

    for (const auto& child_tensor : child_tensors) {
        std::cout << "Contracting with child tensor dimensions: ("
                  << child_tensor.dimension(0) << "x" << child_tensor.dimension(1) << ")" << std::endl;

        Eigen::array<Eigen::IndexPair<int>, 1> contraction_dims = {Eigen::IndexPair<int>(1, 0)};  // Adjust based on tensor structure

        // Check if dimensions match for contraction
        if (parent_tensor.dimension(1) != child_tensor.dimension(0)) {
            throw std::runtime_error("Dimension mismatch detected during contraction");
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

    return toMatrix(parent_tensor);  // Convert back to Eigen::MatrixXcd
}
