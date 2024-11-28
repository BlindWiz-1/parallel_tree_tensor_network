#ifndef ORTHONORMALIZATION_H
#define ORTHONORMALIZATION_H

#include <memory>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

class TNode;  // Forward declaration

// Function declarations
Eigen::MatrixXcd orthonormalizeSVD(const std::shared_ptr<TNode>& node, int i, int d_max);
void contractFactorOnIndex(const std::shared_ptr<TNode>& node, const Eigen::MatrixXcd& factor, int index);
void precontractRoot(const std::shared_ptr<TNode>& node, int site_j, const Eigen::MatrixXcd& factor);

#endif // ORTHONORMALIZATION_H
