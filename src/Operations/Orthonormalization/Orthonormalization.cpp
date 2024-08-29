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

// Function to perform SVD orthonormalization
Tensor orthonormalizeSVD(const Tensor& tensor, int i, int d_max, const std::optional<Tensor>& factor, const std::unordered_map<std::string, int>& leaf_indices) {
    Tensor temp_tensor = tensor;
    if (factor.has_value()) {
        temp_tensor = contractFactorOnIndex(temp_tensor, factor.value(), leaf_indices.at(std::to_string(i)));
    }

    Eigen::Index rows = temp_tensor.rows();
    Eigen::Index cols = temp_tensor.cols();
    temp_tensor = temp_tensor.reshaped(rows * cols, 1); // Flatten the tensor

    Eigen::JacobiSVD svd(temp_tensor, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Tensor u_matrix = svd.matrixU();
    Eigen::VectorXd singular_values = svd.singularValues();
    Tensor v_matrix = svd.matrixV();

    int effective_d_max = std::min(d_max, static_cast<int>(singular_values.size()));
    u_matrix = u_matrix.reshaped(rows, u_matrix.cols()); // Reshape back
    return (singular_values.head(effective_d_max).asDiagonal() * v_matrix.leftCols(effective_d_max).adjoint());
}

// Function to contract a factor on a given index
Tensor contractFactorOnIndex(const Tensor& tensor, const Tensor& factor, int idx) {
    Tensor result_tensor = tensor * factor;
    return result_tensor;
}

// Function to precontract the root node
void precontractRoot(TNode& node, int site_j, const Tensor& factor) {
    assert(node.isRoot());

    // Extract the tensor and temporary factor as Eigen matrices
    Tensor tensor = node.getTensor();
    Tensor tmp_factor = node.getTmpFactor().value();

    auto start_time = std::chrono::high_resolution_clock::now();

    // Perform matrix multiplication using Eigen
    Tensor result_tensor = tensor * tmp_factor;

    // Create an identity matrix using Eigen
    Tensor identity = Tensor::Identity(node.getTmpDim(), node.getTmpDim());

    // Perform additional matrix multiplication and add it to the result
    result_tensor += tmp_factor.block(0, 0, node.getTmpDim(), tmp_factor.cols()) *
                     factor.block(0, site_j, node.getTmpDim(), factor.cols()).adjoint();

    // Update the node tensor with the result
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
