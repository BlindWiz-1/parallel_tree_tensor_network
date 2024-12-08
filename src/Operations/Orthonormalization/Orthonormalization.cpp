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
        auto tensor_tensor = std::get<Eigen::Tensor<std::complex<double>, 3>>(tensor_variant);
        int dim0 = dimensions[0];
        int dim1 = dimensions[1];
        int dim2 = dimensions[2];

        // Map Tensor<3> to MatrixXcd
        tensor_matrix = Eigen::Map<Eigen::MatrixXcd>(
            tensor_tensor.data(), dim0 * dim1, dim2
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
        node->setTensor(U);  // Set truncated U as the new tensor
    } else if (std::holds_alternative<Eigen::Tensor<std::complex<double>, 3>>(tensor_variant)) {
        int dim0 = dimensions[0];
        int dim1 = dimensions[1];
        auto new_tensor_shape = Eigen::array<Eigen::Index, 3>{dim0, dim1, effective_d_max};

        Eigen::Tensor<std::complex<double>, 3> new_tensor(new_tensor_shape);
        Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 2>> tensor_map(
            U.data(), dim0 * dim1, effective_d_max
        );
        new_tensor = tensor_map.reshape(new_tensor_shape);

        node->setTensor(new_tensor);
        std::cout << "Updated tensor shape after SVD orthonormalization: ("
                  << new_tensor.dimension(0) << "x" << new_tensor.dimension(1) << "x" << new_tensor.dimension(2) << ")" << std::endl;
    }

    // Compute and return ΣVᵀ
    Eigen::MatrixXcd factor_result = singular_values.asDiagonal() * V.adjoint();
    printMatrixDimensions(factor_result, "Factor result");
    return factor_result;
}

// Contract factor with tensor
void contractFactorOnIndex(const std::shared_ptr<TNode>& node, const Eigen::MatrixXcd& factor, int idx) {
    auto tensor_variant = node->getTensor();

    if (std::holds_alternative<Eigen::Tensor<std::complex<double>, 3>>(tensor_variant)) {
        auto tensor_tensor = std::get<Eigen::Tensor<std::complex<double>, 3>>(tensor_variant);

        // Convert MatrixXcd to Tensor
        Eigen::Tensor<std::complex<double>, 2> factor_tensor(factor.rows(), factor.cols());

        for (int i = 0; i < factor.rows(); ++i) {
            for (int j = 0; j < factor.cols(); ++j) {
                factor_tensor(i, j) = factor(i, j);
            }
        }

        // Define contraction axes
        Eigen::array<Eigen::IndexPair<int>, 1> contraction_dims = {Eigen::IndexPair<int>(2, 0)};

        // Perform tensor contraction
        Eigen::Tensor<std::complex<double>, 3> result = tensor_tensor.contract(factor_tensor, contraction_dims);

        node->setTensor(result);

    } else if (std::holds_alternative<Eigen::MatrixXcd>(tensor_variant)) {
        auto tensor_matrix = std::get<Eigen::MatrixXcd>(tensor_variant);

        if (tensor_matrix.cols() != factor.rows()) {
            throw std::runtime_error("Dimension mismatch: tensor_matrix.cols() != factor.rows()");
        }

        // Perform matrix multiplication
        Eigen::MatrixXcd result = tensor_matrix * factor;

        node->setTensor(result);
    } else {
        throw std::runtime_error("Unsupported tensor type in contractFactorOnIndex");
    }

    std::cout << "Completed contraction with factor on index: " << idx << std::endl;
}

void precontractRoot(const std::shared_ptr<TNode>& node, int site_j, const Eigen::MatrixXcd& factor) {
    assert(node->isRoot());

    auto tensor_variant = node->getTensor();
    auto tmp_factor_opt = node->getTmpFactor();

    if (!tmp_factor_opt.has_value()) {
        throw std::runtime_error("No tmp factor stored in root node");
    }

    Eigen::MatrixXcd tmp_factor = tmp_factor_opt.value();

    // Log for debugging
    std::cout << "Starting precontractRoot:" << std::endl;
    std::cout << "tmp_factor dimensions: (" << tmp_factor.rows() << "x" << tmp_factor.cols() << ")" << std::endl;
    std::cout << "factor dimensions: (" << factor.rows() << "x" << factor.cols() << ")" << std::endl;

    if (std::holds_alternative<Eigen::Tensor<std::complex<double>, 3>>(tensor_variant)) {
        auto tensor_tensor = std::get<Eigen::Tensor<std::complex<double>, 3>>(tensor_variant);

        int dim0 = tensor_tensor.dimension(0);
        int dim1 = tensor_tensor.dimension(1);
        int dim2 = tensor_tensor.dimension(2);

        // Map Tensor<3> to MatrixXcd
        Eigen::MatrixXcd tensor_matrix = Eigen::Map<Eigen::MatrixXcd>(
            tensor_tensor.data(), dim0 * dim1, dim2
        );

        // Perform contraction
        if (tensor_matrix.cols() != tmp_factor.rows() || tmp_factor.cols() != factor.rows()) {
            throw std::runtime_error("Dimension mismatch during root contraction");
        }
        tensor_matrix = tensor_matrix * tmp_factor * factor;

        // Reshape result back to Tensor<3>
        auto new_shape = Eigen::array<Eigen::Index, 3>{dim0, dim1, factor.cols()};
        Eigen::Tensor<std::complex<double>, 3> result_tensor(new_shape);
        Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 2>> contracted_map(
            tensor_matrix.data(), dim0 * dim1, factor.cols()
        );
        result_tensor = contracted_map.reshape(new_shape);

        node->setTensor(result_tensor);
        std::cout << "Final root tensor dimensions after contraction: ("
                  << result_tensor.dimension(0) << "x"
                  << result_tensor.dimension(1) << "x"
                  << result_tensor.dimension(2) << ")" << std::endl;
    } else if (std::holds_alternative<Eigen::MatrixXcd>(tensor_variant)) {
        // Handle MatrixXcd directly
        auto tensor_matrix = std::get<Eigen::MatrixXcd>(tensor_variant);

        if (tensor_matrix.cols() != tmp_factor.rows() || tmp_factor.cols() != factor.rows()) {
            throw std::runtime_error("Dimension mismatch during root contraction");
        }

        tensor_matrix = tensor_matrix * tmp_factor * factor;
        node->setTensor(tensor_matrix);

        std::cout << "Final root tensor_matrix dimensions: ("
                  << tensor_matrix.rows() << "x" << tensor_matrix.cols() << ")" << std::endl;
    } else {
        throw std::runtime_error("Unsupported tensor format in root node");
    }

    // Clear temporary data
    node->setTmpFactor(std::nullopt);
    node->setTmpDim(0);
    node->setTmpIndex(-1);
}
