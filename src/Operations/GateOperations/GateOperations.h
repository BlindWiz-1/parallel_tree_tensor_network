#ifndef GATE_OPERATIONS_H
#define GATE_OPERATIONS_H

#include <tuple>
#include "../../TTNCircuitSim/TTN/TTN.h"

class GateOperations {
public:
    static void applySingleParticleGate(TTN& psi, const Eigen::MatrixXcd& gate_matrix, int site);
    static std::tuple<std::vector<Eigen::MatrixXcd>, Eigen::VectorXd, std::vector<Eigen::MatrixXcd>> decomposeTwoParticleGate(const Eigen::MatrixXcd& gate_matrix, int local_dimension = 2);
    static void applyTwoParticleGate(TTN& psi, const Eigen::MatrixXcd& gate_matrix, int site_i, int site_j);
    static void applyCircuit(TTN& psi, const Circuit& circ);
};

#endif // GATE_OPERATIONS_H
