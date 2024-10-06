#include "Orthonormalization.h"
#include <Eigen/Dense>
#include <iostream>
#include <cassert>
#include <numeric>
#include <unordered_map>
#include <optional>
#include "../TTNContract/TTNContract.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>
#include "xtensor-blas/xlinalg.hpp"
#include <unsupported/Eigen/CXX11/Tensor>

// Function to perform QR orthonormalization
Tensor orthonormalizeQR(const Tensor& tensor, const std::optional<Tensor>& factor, const std::unordered_map<std::string, int>& leaf_indices, int i) {
    Tensor temp_tensor = tensor;
    if (factor.has_value()) {
        temp_tensor = contractFactorOnIndex(temp_tensor, factor.value(), leaf_indices.at(std::to_string(i)));
    }

    Eigen::Index rows = temp_tensor.rows();
    Eigen::Index cols = temp_tensor.cols();
    temp_tensor = temp_tensor.reshaped(rows * cols, 1); // Flatten the tensor
    Eigen::HouseholderQR<Tensor> qr(temp_tensor);
    Tensor

    q_matrix = qr.householderQ() * Tensor::Identity(rows, cols),
    r_matrix = qr.matrixQR().triangularView<Eigen::Upper>();

    Tensor reshaped_q_matrix = q_matrix.reshaped(rows, q_matrix.cols());
    //TODO: tensor = q_matrix.reshaped(rows, q_matrix.cols()); // Reshape back

    return r_matrix;
}

Tensor orthonormalizeSVD(const Tensor& tensor, int i, int d_max, const std::optional<Tensor>& factor, const std::unordered_map<std::string, int>& leaf_indices) {
    Tensor temp_tensor = tensor;
    if (factor.has_value()) {
        assert(factor.value().cols() == temp_tensor.rows() && "Factor and tensor dimensions must match for contraction");
        temp_tensor = contractFactorOnIndex(temp_tensor, factor.value(), leaf_indices.at(std::to_string(i)));
    }

    Eigen::Index rows = temp_tensor.rows();
    Eigen::Index cols = temp_tensor.cols();

    temp_tensor = temp_tensor.reshaped(rows * cols, 1);

    Eigen::JacobiSVD svd(temp_tensor, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Tensor u_matrix = svd.matrixU();
    Eigen::VectorXd singular_values = svd.singularValues();
    Tensor v_matrix = svd.matrixV();

    int effective_d_max = std::min(d_max, static_cast<int>(singular_values.size()));
    u_matrix = u_matrix.reshaped(rows, u_matrix.cols());

    return (singular_values.head(effective_d_max).asDiagonal() * v_matrix.leftCols(effective_d_max).adjoint());
}

Tensor contractFactorOnIndex(const Tensor& tensor, const Tensor& factor, int idx) {
    // Ensure dimensions match for multiplication
    assert(tensor.cols() == factor.rows() && "Tensor and factor dimensions must match for multiplication");

    Tensor result_tensor = tensor * factor;
    return result_tensor;
}

void precontractRoot(TNode& node, int site_j, const Tensor& factor) {
    assert(node.isRoot());

    Tensor tensor = node.getTensor();
    Tensor tmp_factor = node.getTmpFactor().value();
    int tmp_dim = node.getTmpDim();

    // Ensure dimensions are valid for the operation
    assert(tmp_factor.cols() >= tmp_dim && factor.cols() >= tmp_dim);

    // Create identity matrix of size (tmp_dim, tmp_dim)
    Tensor identity = Tensor::Identity(tmp_dim, tmp_dim);

    // Apply the contraction using matrix operations and blocks
    // Assumes that the tensor dimensions are compatible with tmp_factor and factor
    Tensor result_tensor = tensor;

    result_tensor += tmp_factor.leftCols(tmp_dim) * identity * factor.leftCols(tmp_dim).adjoint();

    // Set the resulting tensor back to the node
    node.setTensor(result_tensor);

    // Reset temporary parameters
    node.setTmpDim(0);
    node.setTmpFactor(std::nullopt);
    node.setTmpIndex(-1);

    std::cout << "Root contraction completed." << std::endl;
}
