/* #ifndef TEST_TTN_H
#define TEST_TTN_H

#include <gtest/gtest.h>
#include "../TTN/TTN.h"
#include "../../Circuits/Circuit/Circuit.h"
#include "../../Circuits/CircuitGate/CircuitGate.h"
#include "../../Circuits/QuantumGate/QuantumGate.h"
#include "../../Structure/SNode/SNode.h"
#include "../TNode/TNode.h"
#include "../PseudoTNode/PseudoTNode.h"
#include "../../Operations/GateOperations/GateOperations.h"
#include <Eigen/Dense>

// Test fixture for TTN tests
class TestTTN : public ::testing::Test {
protected:
    Circuit circ;

    // This function is automatically called before each test in this class
    void SetUp() override;

    // Other protected members can be added here if needed
};

// Function declarations for the tests
TEST_F(TestTTN, BasisState);
TEST_F(TestTTN, AsVector);
TEST_F(TestTTN, BondData);
TEST_F(TestTTN, MaxLeaves);
TEST_F(TestTTN, Orthonormalize);

// Other test classes for helper functions
class TestTTNHelpers : public ::testing::Test {
protected:
    // Setup function if needed
    void SetUp() override;
};

// Helper test function declarations
TEST_F(TestTTNHelpers, IsSquareIdentity);
TEST_F(TestTTNHelpers, FindBestStructure);

#endif // TEST_TTN_H
*/