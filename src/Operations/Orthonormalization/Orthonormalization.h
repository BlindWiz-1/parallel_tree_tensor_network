//
// Created by Denado Rabeli on 7/28/24.
//

#ifndef ORTHONORMALIZATION_H
#define ORTHONORMALIZATION_H

#include <Eigen/Dense>
#include <optional>
#include "../../TTNCircuitSim/TNode/TNode.h"
#include "../../TTNCircuitSim/PseudoTNode/PseudoTNode.h"

// Function to perform QR orthonormalization
Tensor orthonormalizeQR(const Tensor& tensor, const std::optional<Tensor>& factor, const std::unordered_map<std::string, int>& leaf_indices, int i);

// Function to perform SVD orthonormalization
Tensor orthonormalizeSVD(Tensor tensor, int i, int d_max, const std::optional<Tensor>& factor, const std::unordered_map<std::string, int>& leaf_indices);

// Function to contract a factor on a given index
Tensor contractFactorOnIndex(const Tensor& tensor, const Tensor& factor, int idx);

// Function to precontract the root node
void precontractRoot(TNode& node, int site_j, const Tensor& factor);

#endif // ORTHONORMALIZATION_H
