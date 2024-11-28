#include "Orthonormalization.h"
#include "../../TTNCircuitSim/TNode/TNode.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <cassert>
#include <variant>

// Utility to print Eigen matrix dimensions
template <typename T>
void printMatrixDimensions(const T& matrix, const std::string& name) {
    std::cout << name << " dimensions: (" << matrix.rows() << "x" << matrix.cols() << ")" << std::endl;
}

// SVD-based orthonormalization with added debug logging
Eigen::MatrixXcd orthonormalizeSVD(const std::shared_ptr<TNode>& node, int i, int d_max) {
    // Extract tensor and dimensions
    auto tensor_variant = node->getTensor();
    auto dimensions = node->getTensorDimensions();
    Eigen::MatrixXcd tensor_matrix;

    // Handle tensor types (MatrixXcd or Tensor<3>)
    if (std::holds_alternative<Eigen::MatrixXcd>(tensor_variant)) {
        tensor_matrix = std::get<Eigen::MatrixXcd>(tensor_variant);
        printMatrixDimensions(tensor_matrix, "Node tensor (MatrixXcd)");
    } else if (std::holds_alternative<Eigen::Tensor<std::complex<double>, 3>>(tensor_variant)) {
        // Map Tensor<3> to MatrixXcd without manual loops
        auto tensor_tensor = std::get<Eigen::Tensor<std::complex<double>, 3>>(tensor_variant);
        int dim0 = dimensions[0];
        int dim1 = dimensions[1];
        int dim2 = dimensions[2];

        tensor_matrix = Eigen::Map<Eigen::MatrixXcd>(
            tensor_tensor.data(),
            dim0 * dim1, dim2
        );
        printMatrixDimensions(tensor_matrix, "Node tensor (Tensor)");
    } else {
        throw std::runtime_error("Unknown tensor type in orthonormalizeSVD");
    }

    // Perform SVD
    std::cout << "Performing SVD..." << std::endl;
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(tensor_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXcd U = svd.matrixU();
    Eigen::VectorXd singular_values = svd.singularValues();
    Eigen::MatrixXcd V = svd.matrixV();

    printMatrixDimensions(U, "U before filtering");
    printMatrixDimensions(V, "V before filtering");

    // Truncate to d_max if necessary
    int effective_d_max = (d_max > 0) ? std::min(d_max, static_cast<int>(singular_values.size())) : singular_values.size();
    U = U.leftCols(effective_d_max);
    V = V.leftCols(effective_d_max);
    singular_values.conservativeResize(effective_d_max);

    printMatrixDimensions(U, "U after filtering");
    printMatrixDimensions(V, "V after filtering");

    // Update node tensor
    if (std::holds_alternative<Eigen::MatrixXcd>(tensor_variant)) {
        // Directly set the truncated U matrix
        node->setTensor(U);
    } else if (std::holds_alternative<Eigen::Tensor<std::complex<double>, 3>>(tensor_variant)) {
        // Reshape truncated U matrix into Tensor<3>
        auto new_tensor_shape = Eigen::array<Eigen::Index, 3>{dimensions[0], dimensions[1], effective_d_max};
        Eigen::Tensor<std::complex<double>, 3> new_tensor(new_tensor_shape);

        Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 2>> tensor_map(
            U.data(), dimensions[0] * dimensions[1], effective_d_max
        );
        new_tensor = tensor_map.reshape(new_tensor_shape);

        node->setTensor(new_tensor);
    }

    // Compute factor result (ΣVᵀ)
    Eigen::MatrixXcd factor_result = singular_values.asDiagonal() * V.adjoint();
    printMatrixDimensions(factor_result, "Factor result");

    return factor_result;
}

// Contract factor with tensor
void contractFactorOnIndex(const std::shared_ptr<TNode>& node, const Eigen::MatrixXcd& factor, int index) {
    auto tensor_variant = node->getTensor();

    if (std::holds_alternative<Eigen::MatrixXcd>(tensor_variant)) {
        auto tensor_matrix = std::get<Eigen::MatrixXcd>(tensor_variant);
        printMatrixDimensions(tensor_matrix, "Tensor (MatrixXcd) before contraction");
        tensor_matrix = tensor_matrix * factor;
        node->setTensor(tensor_matrix);
    } else if (std::holds_alternative<Eigen::Tensor<std::complex<double>, 3>>(tensor_variant)) {
        auto tensor_tensor = std::get<Eigen::Tensor<std::complex<double>, 3>>(tensor_variant);

        int dim0 = tensor_tensor.dimension(0);
        int dim1 = tensor_tensor.dimension(1);
        int dim2 = tensor_tensor.dimension(2);

        Eigen::MatrixXcd tensor_matrix(dim0 * dim1, dim2);
        const std::complex<double>* data_ptr = tensor_tensor.data();
        for (Eigen::Index idx = 0; idx < tensor_matrix.size(); ++idx) {
            tensor_matrix(idx / dim2, idx % dim2) = data_ptr[idx];
        }

        printMatrixDimensions(tensor_matrix, "Tensor before contraction");
        tensor_matrix = tensor_matrix * factor;

        Eigen::Tensor<std::complex<double>, 3> new_tensor(dim0, dim1, tensor_matrix.cols());
        for (int idx = 0; idx < tensor_matrix.rows(); ++idx) {
            int i0 = idx / dim1;
            int i1 = idx % dim1;
            for (int j = 0; j < tensor_matrix.cols(); ++j) {
                new_tensor(i0, i1, j) = tensor_matrix(idx, j);
            }
        }
        node->setTensor(new_tensor);
    }
    std::cout << "Completed contraction with factor on index: " << index << std::endl;
}

// Handle root contraction with debug logging
void precontractRoot(const std::shared_ptr<TNode>& node, int site_j, const Eigen::MatrixXcd& factor) {
    assert(node->isRoot());

    auto tensor_variant = node->getTensor();
    auto tmp_factor_opt = node->getTmpFactor();
    if (!tmp_factor_opt.has_value()) throw std::runtime_error("No tmp factor stored in root node");

    Eigen::MatrixXcd tmp_factor = tmp_factor_opt.value();
    if (std::holds_alternative<Eigen::MatrixXcd>(tensor_variant)) {
        auto tensor_matrix = std::get<Eigen::MatrixXcd>(tensor_variant);
        printMatrixDimensions(tensor_matrix, "Root tensor (MatrixXcd) before contraction");
        tensor_matrix = tensor_matrix * tmp_factor * factor;
        node->setTensor(tensor_matrix);
    } else if (std::holds_alternative<Eigen::Tensor<std::complex<double>, 3>>(tensor_variant)) {
        auto tensor_tensor = std::get<Eigen::Tensor<std::complex<double>, 3>>(tensor_variant);

        int dim0 = tensor_tensor.dimension(0);
        int dim1 = tensor_tensor.dimension(1);
        int dim2 = tensor_tensor.dimension(2);

        Eigen::MatrixXcd tensor_matrix(dim0 * dim1, dim2);
        const std::complex<double>* data_ptr = tensor_tensor.data();
        for (Eigen::Index idx = 0; idx < tensor_matrix.size(); ++idx) {
            tensor_matrix(idx / dim2, idx % dim2) = data_ptr[idx];
        }

        tensor_matrix = tensor_matrix * tmp_factor * factor;
        Eigen::Tensor<std::complex<double>, 3> new_tensor(dim0, dim1, tensor_matrix.cols());
        for (int idx = 0; idx < tensor_matrix.rows(); ++idx) {
            int i0 = idx / dim1;
            int i1 = idx % dim1;
            for (int j = 0; j < tensor_matrix.cols(); ++j) {
                new_tensor(i0, i1, j) = tensor_matrix(idx, j);
            }
        }
        node->setTensor(new_tensor);
    }
    std::cout << "Completed root contraction." << std::endl;

    node->setTmpFactor(std::nullopt);
    node->setTmpDim(0);
    node->setTmpIndex(-1);
}
