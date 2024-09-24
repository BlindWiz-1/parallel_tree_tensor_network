#ifndef TTNSAMPLING_H
#define TTNSAMPLING_H

#include <memory>
#include "../../TTNCircuitSim/TNode/TNode.h"
#include <Eigen/Dense>
#include <vector>

using Tensor = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>;

class TTNSampling {
public:
    static Tensor sample(std::shared_ptr<TNode> root, double nrm);
    static std::vector<int> sampleAndContract(std::shared_ptr<TNode> node, Tensor& current_state);

private:
    static int sampleQubit(const Tensor& tensor, int shots = 1024);
    static int sampleOnce(const std::vector<double>& state_probabilities);
};

#endif // TTNSAMPLING_H
