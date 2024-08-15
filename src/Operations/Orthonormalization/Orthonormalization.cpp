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
#include <xtensor/xview.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xoperation.hpp>
#include "xtensor-blas/xlinalg.hpp"

// Function to perform QR orthonormalization
Tensor orthonormalizeQR(const Tensor& tensor, const std::optional<Tensor>& factor, const std::unordered_map<std::string, int>& leaf_indices, int i) {
    Tensor temp_tensor = tensor;
    if (factor.has_value()) {
        temp_tensor = contractFactorOnIndex(temp_tensor, factor.value(), leaf_indices.at(std::to_string(i)));
    }

    Eigen::Index rows = temp_tensor.rows();
    Eigen::Index cols = temp_tensor.cols();
    temp_tensor = temp_tensor.reshaped(rows * cols, 1); // Flatten the tensor
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(temp_tensor);
    Eigen::MatrixXd

    q_matrix = qr.householderQ() * Eigen::MatrixXd::Identity(rows, cols),
    r_matrix = qr.matrixQR().triangularView<Eigen::Upper>();

    q_matrix = q_matrix.reshaped(rows, q_matrix.cols()); // Reshape back
    return r_matrix;
}

// Function to perform SVD orthonormalization
Tensor orthonormalizeSVD(const Tensor& tensor, int i, double tol, int d_max, const std::optional<Tensor>& factor, const std::unordered_map<std::string, int>& leaf_indices) {
    Tensor temp_tensor = tensor;
    if (factor.has_value()) {
        temp_tensor = contractFactorOnIndex(temp_tensor, factor.value(), leaf_indices.at(std::to_string(i)));
    }

    Eigen::Index rows = temp_tensor.rows();
    Eigen::Index cols = temp_tensor.cols();
    temp_tensor = temp_tensor.reshaped(rows * cols, 1); // Flatten the tensor

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(temp_tensor, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Tensor u_matrix = svd.matrixU();
    Tensor singular_values = svd.singularValues();
    Tensor v_matrix = svd.matrixV();

    int effective_d_max = std::min(d_max, static_cast<int>(singular_values.size()));
    u_matrix = u_matrix.reshaped(rows, u_matrix.cols()); // Reshape back
    return (singular_values.head(effective_d_max).asDiagonal() * v_matrix.leftCols(effective_d_max).adjoint());
}

// Function to contract a factor on a given index
Tensor contractFactorOnIndex(const Tensor& tensor, const Tensor& factor, int idx) {
    // Convert the input tensors to xt::xarray
    xt::xarray<std::complex<double>> xt_tensor = xt::adapt(tensor);
    xt::xarray<std::complex<double>> xt_factor = xt::adapt(factor);

    // Perform the contraction using xtensor
    xt::xarray<std::complex<double>> result = xt::zeros<std::complex<double>>({xt_tensor.shape(0), xt_factor.shape(1)});

    for (size_t i = 0; i < xt_tensor.shape(0); ++i) {
        for (size_t j = 0; j < xt_factor.shape(1); ++j) {
            for (size_t k = 0; k < xt_tensor.shape(1); ++k) {
                result(i, j) += xt_tensor(i, k) * xt_factor(k, j);
            }
        }
    }

    // Convert the result back to Eigen::Matrix
    Tensor result_tensor = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>>(result.data(), result.shape(0), result.shape(1));

    return result_tensor;
}


// Function to precontract the root node

void precontractRoot(TNode& node, int site_j, const Tensor& factor) {
    assert(node.isRoot());

    // Convert Tensor to xt::xarray
    xt::xarray<std::complex<double>> tensor = xt::adapt(node.getTensor());
    int site_i = node.getLeafIndices().at(std::to_string(node.getTmpIndex()));
    site_j = node.getLeafIndices().at(std::to_string(site_j));

    auto start_time = std::chrono::high_resolution_clock::now();

    // Convert tmp_factor to xt::xarray
    xt::xarray<std::complex<double>> tmp_factor = xt::adapt(node.getTmpFactor().value());

    // Contraction using xtensor-blas
    xt::xarray<std::complex<double>> result = xt::linalg::dot(tensor, tmp_factor);

    // Add identity contraction
    xt::xarray<std::complex<double>> identity = xt::eye<std::complex<double>>(node.getTmpDim());
    result += xt::linalg::dot(xt::view(tmp_factor, xt::range(0, node.getTmpDim()), xt::all()),
                             xt::view(xt::adapt(factor), xt::range(0, node.getTmpDim()), xt::range(site_j, factor.cols())));

    // conversion back to Eigen::Matrix from xt::xarray
    Tensor result_tensor = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>>(result.data(), result.shape(0), result.shape(1));

    // Update the node tensor
    node.setTensor(result_tensor);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Root contraction took " << elapsed.count() << " seconds" << std::endl;

    // Reset root temporary parameters
    node.setTmpDim(0);
    node.setTmpFactor(std::nullopt); // Reset to an empty optional
    node.setTmpIndex(-1);
}

// Function for pseudo-orthonormalization
int pseudoOrthonormalize(PseudoTNode& node, int i, std::optional<int> factor) {
    auto shape = node.getShape();

    if (factor.has_value()) {
        i = node.getLeafIndices().at(std::to_string(i));
        shape.insert(shape.begin() + i, factor.value());
    }

    int product = std::accumulate(shape.begin(), shape.end() - 1, 1, std::multiplies<int>());

    if (static_cast<double>(product) > shape.back()) {
        node.setShape(shape);
        return shape.back();
    }

    if (!node.isRoot()) {
        shape.resize(shape.size() - 1);
        shape.push_back(product);
    } else {
        node.setShape(shape);
    }

    return shape.back();
}
