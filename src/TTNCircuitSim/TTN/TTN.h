#ifndef TTN_H
#define TTN_H

#include "../TNode/TNode.h"
#include <vector>

class TTN {
public:
    TTN(int num_qubits, int local_dim);

    void applyCircuit(const std::vector<std::shared_ptr<TNode>>& circuit);
    void display() const;

private:
    std::shared_ptr<TNode> root_;
    int num_qubits_;
    int local_dim_;
};

#endif // TTN_H
