/*
#include <gtest/gtest.h>
#include "../TTN/TTN.h"
#include "../../Circuits/Circuit/Circuit.h"
#include "../../Circuits/CircuitGate/CircuitGate.h"
#include "../../Circuits/QuantumGate/QuantumGate.h"
#include "../../Structure/SNode/SNode.h"
#include "../TNode/TNode.h"
#include "../PseudoTNode/PseudoTNode.h"
#include "../../Operations/GateOperations/GateOperations.h"
#include "TestTTN.h"
#include <Eigen/Dense>

class TestTTN : public ::testing::Test {
protected:
    Circuit circ;

    void SetUp() override {
        circ = Circuit(3, 2);
        circ.appendGate(CircuitGate(QuantumGate::H, {0}));
        circ.appendGate(CircuitGate(QuantumGate::CNOT, {0, 1}));
        circ.appendGate(CircuitGate(QuantumGate::CNOT, {0, 2}));
    }
};

TEST_F(TestTTN, TestBasisState) {
    auto structure = std::make_shared<SNode>("root", std::vector<std::shared_ptr<SNode>>{std::make_shared<SNode>(0), std::make_shared<SNode>(2), std::make_shared<SNode>(1)});
    auto psi = TTN::basisState(2, {0, 0, 0}, structure, circ, 100, true);
    GateOperations::applyCircuit(psi, circ);

    auto target = std::make_shared<PseudoTNode>(
        "root", std::vector<int>{2, 2, 2, 1},
        std::vector<std::shared_ptr<PseudoTNode>>{
            std::make_shared<PseudoTNode>(0, std::vector<int>{2, 2}),
            std::make_shared<PseudoTNode>(2, std::vector<int>{2, 2}),
            std::make_shared<PseudoTNode>(1, std::vector<int>{2, 2})
        },
        std::unordered_map<int, int>{{0, 0}, {1, 2}, {2, 1}}
    );
    target->setLocalDim(1);

    EXPECT_TRUE(pseudoEqual(psi->getTNodeRoot(), target));

    psi = TTN::basisState(2, {0, 0, 0}, nullptr, circ, 100, false, 1);
    GateOperations::applyCircuit(psi, circ, true);

    auto target_tnode = std::make_shared<TNode>(
        "root", Tensor<std::complex<double>, 3>::Constant(1),
        std::vector<std::shared_ptr<TNode>>{
            std::make_shared<TNode>(
                "0.0", Eigen::Tensor<std::complex<double>, 3>::Constant(1),
                std::vector<std::shared_ptr<TNode>>{
                    std::make_shared<TNode>(0, Eigen::MatrixXcd::Identity(2, 1)),
                    std::make_shared<TNode>(1, Eigen::MatrixXcd::Identity(2, 1))
                },
                std::unordered_map<int, int>{{0, 0}, {1, 1}}
            ),
            std::make_shared<TNode>(2, Eigen::MatrixXcd::Identity(2, 1))
        },
        std::unordered_map<int, int>{{0, 0}, {1, 0}, {2, 1}}
    );

    EXPECT_NEAR(psi->nrm(), 1 / std::sqrt(2), 1e-9);
    EXPECT_TRUE(nodeEqual(psi->getTNodeRoot(), target_tnode));
}

TEST_F(TestTTN, TestAsVector) {
    auto psi = TTN::basisState(2, {0, 0, 0}, nullptr, circ);
    EXPECT_TRUE(psi->asVector().isApprox(Eigen::VectorXd::Unit(8, 0)));

    GateOperations::applyCircuit(psi, circ, true);
    Eigen::VectorXd expected_vector(8);
    expected_vector << 1, 0, 0, 0, 0, 0, 0, 1;
    EXPECT_TRUE(psi->asVector().isApprox(expected_vector / std::sqrt(2)));
}

TEST_F(TestTTN, TestBondData) {
    auto psi = TTN::basisState(2, {0, 0, 0}, nullptr, circ);
    EXPECT_EQ(psi->bondData(), std::make_pair(8.0f, 2));

    GateOperations::applyCircuit(psi, circ, true);
    EXPECT_EQ(psi->bondData(), std::make_pair(24.0f, 2));
}

TEST_F(TestTTN, TestMaxLeaves) {
    auto psi = TTN::basisState(2, {0, 0, 0}, nullptr, circ);
    EXPECT_EQ(psi->maxLeaves(), 2);

    auto structure = std::make_shared<SNode>("root", std::vector<std::shared_ptr<SNode>>{std::make_shared<SNode>(0), std::make_shared<SNode>(1), std::make_shared<SNode>(2)});
    psi = TTN::basisState(2, {0, 0, 0}, structure, nullptr);
    EXPECT_EQ(psi->maxLeaves(), 1);
}

TEST_F(TestTTN, TestOrthonormalize) {
    Eigen::MatrixXcd target(2, 2);
    target << 1, 0, 0, -1;
    target /= std::sqrt(std::sqrt(2));

    auto root = std::make_shared<TNode>(
        "root", Eigen::Tensor<std::complex<double>, 3>::Constant(1),
        std::vector<std::shared_ptr<TNode>>{
            std::make_shared<TNode>(0, target),
            std::make_shared<TNode>(1, target)
        },
        std::unordered_map<int, int>{{0, 0}, {1, 1}}
    );
    root->setTmpDim(2);
    root->setTmpIndex(0);

    auto psi = std::make_shared<TTN>(2, root, 2, false, false);
    psi->orthonormalize(0, 1, true);

    Eigen::MatrixXcd identity = Eigen::MatrixXcd::Identity(2, 2);
    EXPECT_TRUE(psi->getTNodeRoot()->getChildren()[0]->getTensor().isApprox(identity));
    EXPECT_TRUE(psi->getTNodeRoot()->getChildren()[1]->getTensor().isApprox(identity));

    Eigen::Tensor<std::complex<double>, 3> expected_tensor(2, 2, 1);
    expected_tensor.setConstant(1.0 / std::sqrt(2));
    EXPECT_TRUE(psi->getTNodeRoot()->getTensor().isApprox(expected_tensor));
    EXPECT_EQ(psi->getTNodeRoot()->getTmpDim(), 0);
    EXPECT_EQ(psi->getTNodeRoot()->getTmpIndex(), -1);
    EXPECT_EQ(psi->getTNodeRoot()->getTmpFactor(), nullptr);
}

class TestTTNHelpers : public ::testing::Test {};

TEST_F(TestTTNHelpers, TestIsSquareIdentity) {
    EXPECT_TRUE(isSquareIdentity(Eigen::MatrixXd::Identity(4, 4)));
    EXPECT_FALSE(isSquareIdentity(Eigen::MatrixXd::Ones(4, 4)));
    EXPECT_FALSE(isSquareIdentity(nullptr)); // adjust as needed
    EXPECT_FALSE(isSquareIdentity(2)); // adjust as needed
    EXPECT_FALSE(isSquareIdentity(Eigen::MatrixXd::Random(2, 4)));
}

TEST_F(TestTTNHelpers, TestFindBestStructure) {
    std::unordered_map<int, _Structure> structures = {
        {0, _Structure(2, 4, 2, 1)},
        {1, _Structure(3, 4, 3, 2)},
        {2, _Structure(2, 2, 2, 3)}
    };

    EXPECT_EQ(findBestStructure(structures), 1);
    EXPECT_EQ(findBestStructure(structures, true), 2);
    EXPECT_EQ(findBestStructure(structures, false, 2), 2);

    structures.erase(2);
    EXPECT_EQ(findBestStructure(structures, false, 2), 1);
}
*/