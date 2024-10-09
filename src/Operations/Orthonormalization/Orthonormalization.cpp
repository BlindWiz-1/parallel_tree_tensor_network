#include "Orthonormalization.h"
#include <Eigen/Dense>
#include <iostream>
#include <cassert>
#include <numeric>
#include <unordered_map>
#include <optional>
#include "../TTNContract/TTNContract.h"
#include <unsupported/Eigen/CXX11/Tensor>

Eigen::Tensor<std::complex<double>, 2> tensorMap(const Eigen::MatrixXcd& matrix) {
    return Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 2>>(const_cast<std::complex<double>*>(matrix.data()), matrix.rows(), matrix.cols());
}

// Helper function to map a tensor back to a matrix
Eigen::MatrixXcd matrixMap(const Eigen::Tensor<std::complex<double>, 2>& tensor) {
    return Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>>(tensor.data(), tensor.dimension(0), tensor.dimension(1));
}


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

Tensor orthonormalizeSVD(Tensor tensor, int i, int d_max, const std::optional<Tensor>& factor, const std::unordered_map<std::string, int>& leaf_indices) {

    if (factor.has_value()) {
        assert(factor.value().cols() == tensor.rows() && "Factor and tensor dimensions must match for contraction");
        tensor = contractFactorOnIndex(tensor, factor.value(), leaf_indices.at(std::to_string(i)));
    }

    // Reshape the tensor for SVD
    Eigen::Index rows = tensor.rows();
    Eigen::Index cols = tensor.cols();
    Tensor reshaped_tensor = tensor.reshaped(rows * cols, 1);  // Flatten into 2D matrix

    // Perform SVD
    Eigen::JacobiSVD<Tensor> svd(reshaped_tensor, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Tensor u_matrix = svd.matrixU();
    Eigen::VectorXd singular_values = svd.singularValues();
    Tensor v_matrix = svd.matrixV();

    // Truncate if necessary
    int effective_d_max = std::min(d_max, static_cast<int>(singular_values.size()));
    u_matrix.conservativeResize(Eigen::NoChange, effective_d_max);
    v_matrix.conservativeResize(effective_d_max, Eigen::NoChange);
    singular_values.conservativeResize(effective_d_max);

    // Reshape U back to the original dimensions of the tensor
    u_matrix.resize(rows, effective_d_max);

    // Set the current node tensor to U and return S * V
    Tensor result = singular_values.asDiagonal() * v_matrix.adjoint();

    return result;
}

// Function to perform contraction on an index of the tensor
Tensor contractFactorOnIndex(const Tensor& tensor, const Tensor& factor, int index) {
    Eigen::Tensor<std::complex<double>, 2> tensor_eigen = tensorMap(tensor);
    Eigen::Tensor<std::complex<double>, 2> factor_eigen = tensorMap(factor);

    // Perform contraction on the specified index
    Eigen::array<Eigen::IndexPair<int>, 1> contraction_dims = {Eigen::IndexPair<int>(1, 0)};
    Eigen::Tensor<std::complex<double>, 2> result_tensor = tensor_eigen.contract(factor_eigen, contraction_dims);

    // Convert back to Eigen::MatrixXcd
    return matrixMap(result_tensor);
}

void precontractRoot(TNode& node, int site_j, const Tensor& factor) {
    assert(node.isRoot());

    // Map node tensor and factors to Eigen tensors
    Eigen::Tensor<std::complex<double>, 2> tensor = tensorMap(node.getTensor());
    Eigen::Tensor<std::complex<double>, 2> tmp_factor = tensorMap(node.getTmpFactor().value());
    Eigen::Tensor<std::complex<double>, 2> factor_tensor = tensorMap(factor);

    // Extract and validate temporary dimension
    int tmp_dim = node.getTmpDim();
    assert(tmp_dim > 0 && "Temporary dimension must be greater than 0");

    // Ensure leaf indices are correctly set
    int site_i = node.getLeafIndices().at(std::to_string(node.getTmpIndex()));
    site_j = node.getLeafIndices().at(std::to_string(site_j));
    std::cout << "Site index i: " << site_i << ", Site index j: " << site_j << std::endl;

    // Print initial tensor dimensions for debugging
    std::cout << "Initial tensor dimensions: (" << tensor.dimension(0) << "x" << tensor.dimension(1) << ")" << std::endl;
    std::cout << "Tmp factor dimensions: (" << tmp_factor.dimension(0) << "x" << tmp_factor.dimension(1) << ")" << std::endl;
    std::cout << "Factor dimensions: (" << factor_tensor.dimension(0) << "x" << factor_tensor.dimension(1) << ")" << std::endl;

    // Reshape tmp_factor and factor for contraction
    Eigen::array<Eigen::Index, 3> reshape_dims = {tmp_factor.dimension(0), tmp_dim, -1};
    Eigen::Tensor<std::complex<double>, 3> reshaped_tmp_factor = tmp_factor.reshape(reshape_dims);
    Eigen::Tensor<std::complex<double>, 3> reshaped_factor = factor_tensor.reshape(reshape_dims);

    std::cout << "Reshaped tmp_factor dimensions: (" << reshaped_tmp_factor.dimension(0) << ", " << reshaped_tmp_factor.dimension(1) << ", " << reshaped_tmp_factor.dimension(2) << ")" << std::endl;
    std::cout << "Reshaped factor dimensions: (" << reshaped_factor.dimension(0) << ", " << reshaped_factor.dimension(1) << ", " << reshaped_factor.dimension(2) << ")" << std::endl;

    // Prepare contraction dimensions and perform contraction
    Eigen::array<Eigen::IndexPair<int>, 1> contraction_dims = {Eigen::IndexPair<int>(1, 0)};
    Eigen::Tensor<std::complex<double>, 3> contracted_tensor = tensor.contract(reshaped_tmp_factor, contraction_dims);
    contracted_tensor = contracted_tensor.contract(reshaped_factor, contraction_dims);

    // Reshape the contracted tensor back to a 2D matrix
    Eigen::Tensor<std::complex<double>, 2> reshaped_result = contracted_tensor.reshape(Eigen::array<Eigen::Index, 2>{tensor.dimension(0), factor_tensor.dimension(1)});
    std::cout << "Result tensor dimensions after contraction: (" << reshaped_result.dimension(0) << "x" << reshaped_result.dimension(1) << ")" << std::endl;

    // Convert back to MatrixXcd for node tensor
    Eigen::MatrixXcd final_matrix = matrixMap(reshaped_result);
    node.setTensor(final_matrix);

    // Log the new tensor dimensions
    std::cout << "Root tensor dimensions after contraction: (" << final_matrix.rows() << "x" << final_matrix.cols() << ")" << std::endl;

    // Reset temporary root parameters
    node.setTmpDim(0);
    node.setTmpFactor(std::nullopt);
    node.setTmpIndex(-1);

    std::cout << "Root contraction completed and temporary parameters reset.\n" << std::endl;
}
