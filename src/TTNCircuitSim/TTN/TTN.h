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
    TTN(int local_dim, std::shared_ptr<TNode> root, int d_max);

    int localDim() const;
    std::shared_ptr<TNode> getTNodeRoot() const;
    void setTNodeRoot(const std::shared_ptr<TNode>& root);
    int nSites() const;
    static std::shared_ptr<TTN> basisState(int local_dim,
        const std::vector<int>& single_states,
        const std::shared_ptr<SNode>& structure,
        const std::optional<Circuit>& circ = std::nullopt,
        int d_max = 0, bool flat = false);

    Eigen::VectorXd asVector() const;
    std::pair<float, int> bondData() const;
    int maxLeaves() const;
    void orthonormalize(int site_i, int site_j);
    void display() const;

private:
    bool isSquareIdentity(const Eigen::MatrixXcd& factor) const;

    int local_dimension_;
    std::shared_ptr<TNode> root_;
    double nrm_;
    int d_max_;
};

#endif // TTN_H
