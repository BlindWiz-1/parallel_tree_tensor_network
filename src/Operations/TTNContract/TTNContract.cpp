#include "TTNContract.h"
#include <numeric>
#include "../../TTNCircuitSim/TNode/TNode.h"
#include <unsupported/Eigen/CXX11/Tensor>

// Helper function: Converts Eigen::MatrixXcd to Eigen::Tensor
Eigen::Tensor<std::complex<double>, 2> toTensor(const Eigen::MatrixXcd& matrix) {
    return Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 2>>(const_cast<std::complex<double>*>(matrix.data()), matrix.rows(), matrix.cols());
}

// Helper function: Converts Eigen::Tensor to Eigen::MatrixXcd
Eigen::MatrixXcd toMatrix(const Eigen::Tensor<std::complex<double>, 2>& tensor) {
    return Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>>(tensor.data(), tensor.dimension(0), tensor.dimension(1));
}

// Contract function with additional checks and debug information
Tensor contract(const std::shared_ptr<TNode>& node, double nrm) {
    if (node->isLeaf()) {
        std::cout << "Leaf node detected: " << node->getName() << std::endl;
        // Print the dimensions of the leaf tensor
        std::cout << "Leaf tensor dimensions: (" << node->getTensor().rows() << "x" << node->getTensor().cols() << ")" << std::endl;
        return node->getTensor();  // Return leaf tensor as it is
    }

    // Print initial debug information for non-leaf node
    std::cout << "Contracting node: " << node->getName() << std::endl;
    std::cout << "Initial tensor dimensions: (" << node->getTensor().rows() << "x" << node->getTensor().cols() << ")" << std::endl;

    // Convert current node tensor to Eigen::Tensor
    auto parent_tensor = toTensor(node->getTensor());
    int counter = parent_tensor.dimension(1);  // Number of columns as initial counter value
    std::vector<Eigen::Tensor<std::complex<double>, 2>> tensors = {parent_tensor};
    std::vector<Eigen::Index> indices;

    // Track initial indices
    for (Eigen::Index i = 0; i < parent_tensor.dimension(1); ++i) {
        indices.push_back(static_cast<Eigen::Index>(i));
    }

    // Recursively contract each child and add to the tensor list
    for (size_t idx = 0; idx < node->getChildren().size(); ++idx) {
        auto child = node->getChildren()[idx];
        std::cout << "Processing child node: " << child->getName() << std::endl;

        auto child_tensor = toTensor(contract(child, 1));  // Ensure child tensor uses consistent layout
        std::cout << "Child tensor dimensions: (" << child_tensor.dimension(0) << "x" << child_tensor.dimension(1) << ")" << std::endl;

        // Track indices for alignment
        std::vector<Eigen::Index> child_indices(child_tensor.dimension(1) - 1);
        std::iota(child_indices.begin(), child_indices.end(), counter);
        counter += child_tensor.dimension(1) - 1;
        child_indices.emplace_back(idx);

        tensors.push_back(child_tensor);
        indices.insert(indices.end(), child_indices.begin(), child_indices.end());
    }

    if (!node->isRoot()) {
        indices.back() = counter;  // For non-root nodes, adjust parent index for alignment
    } else {
        // Add normalization factor at root node
        Eigen::Tensor<std::complex<double>, 2> norm_tensor(1, 1);
        norm_tensor(0, 0) = std::complex<double>(nrm, 0.0);
        tensors.push_back(norm_tensor);
        indices.emplace_back(parent_tensor.dimension(1) - 1);
    }

    std::cout << "Final indices for contraction: ";
    for (const auto& index : indices) {
        std::cout << index << " ";
    }
    std::cout << std::endl;

    // Initialize result tensor with same layout as parent tensor
    Eigen::Tensor<std::complex<double>, 2> result = tensors[0];

    // Perform tensor contraction using a loop (similar to einsum logic in Python)
    for (size_t i = 1; i < tensors.size(); ++i) {
        // Debug: Log information about the tensors being contracted
        std::cout << "------------------- Contracting Tensor " << i << " -------------------" << std::endl;
        std::cout << "Result tensor dimensions before contraction: (" << result.dimension(0) << "x" << result.dimension(1) << ")" << std::endl;
        std::cout << "Tensor " << i << " dimensions: (" << tensors[i].dimension(0) << "x" << tensors[i].dimension(1) << ")" << std::endl;

        // Check the indices used for contraction
        std::cout << "Indices for contraction: (" << indices[i - 1] << ", " << indices[i] << ")" << std::endl;

        // Align tensor dimensions and perform contraction
        Eigen::array<Eigen::IndexPair<int>, 1> contraction_dims = {Eigen::IndexPair<int>(indices[i - 1], indices[i])};

        // Debug: Log the contraction dimensions
        std::cout << "Contraction dimensions: (" << contraction_dims[0].first << ", " << contraction_dims[0].second << ")" << std::endl;

        // Perform the contraction
        result = result.contract(tensors[i], contraction_dims);

        // Debug: Log result tensor dimensions after contraction
        std::cout << "Result tensor dimensions after contraction: (" << result.dimension(0) << "x" << result.dimension(1) << ")" << std::endl;
        std::cout << "-------------------------------------------------------------" << std::endl;
    }

    // Print final result tensor dimensions
    std::cout << "Final contracted tensor dimensions: (" << result.dimension(0) << "x" << result.dimension(1) << ")" << std::endl;

    // Convert the final Eigen::Tensor result back to Eigen::MatrixXcd
    Eigen::MatrixXcd final_matrix = toMatrix(result);
    return final_matrix;
}
