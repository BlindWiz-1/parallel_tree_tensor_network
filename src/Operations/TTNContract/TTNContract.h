#ifndef TTNCONTRACT_H
#define TTNCONTRACT_H

#include "../../TTNCircuitSim/TNode/TNode.h"
#include <Eigen/Dense>
#include <iostream>

using Tensor = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>;

Tensor contract(const std::shared_ptr<TNode>& node, double nrm, bool enable_gpu = true);

#endif
