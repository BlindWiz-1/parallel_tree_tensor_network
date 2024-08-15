#ifndef TTN_H
#define TTN_H

#include <memory>
#include <vector>
#include <unordered_map>
#include "../TNode/TNode.h"
#include "../../Structure/SNode/SNode.h"
#include "../../Circuits/Circuit/Circuit.h"
#include "../PseudoTNode/PseudoTNode.h"

class TTN {
public:
    // Constructor
    TTN(int local_dim, std::shared_ptr<TNode> root, int d_max = std::numeric_limits<int>::max(), bool enable_gpu = true, bool dry_run = false);
    TTN(int local_dim, std::shared_ptr<PseudoTNode> root, int d_max = std::numeric_limits<int>::max(), bool enable_gpu = true, bool dry_run = false);

    // Accessors
    int localDim() const;
    int nSites() const;
    Eigen::MatrixXd dtype() const;

    // Static method to create a TTN representing a basis state
    static std::shared_ptr<TTN> basisState(int local_dim, const std::vector<int>& single_states, const std::shared_ptr<SNode>& structure = nullptr, const std::optional<Circuit>& circ = std::nullopt, const std::unordered_map<std::string, int>& kwargs = {});

    // Convert TTN to a single vector
    Eigen::VectorXd asVector() const;

    // Get bond data
    std::pair<float, int> bondData() const;

    // Get maximum number of leaves assigned to a subtree of the root
    int maxLeaves() const;

    // Orthonormalize the TTN
    void orthonormalize(int site_i, int site_j, bool compress = false, double tol = 0.0);

    // Apply a circuit to the TTN
    void applyCircuit(const std::vector<std::shared_ptr<TNode>>& circuit);

    // Display the TTN structure
    void display() const;

private:
    int local_dimension_;
    std::shared_ptr<TNode> root_;
    double nrm_;
    int d_max_;
    bool enable_gpu_;
    bool dry_run_;

    // Helper function to check if a matrix is a square identity matrix
    bool isSquareIdentity(const Eigen::MatrixXd& factor) const;
};

#endif // TTN_H
