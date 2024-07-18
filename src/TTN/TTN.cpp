#include "TTN.h"
#include <iostream>

TTN::TTN(int num_qubits, int local_dim)
    : num_qubits_(num_qubits), local_dim_(local_dim) {
    Tensor root_tensor = Tensor::Identity(local_dim_, local_dim_);
    root_ = std::make_shared<TNode>("root", root_tensor);
}

void TTN::applyCircuit(const std::vector<std::shared_ptr<TNode>>& circuit) {
    for (const auto& gate : circuit) {
        std::cout << "Applying gate with shape: " << gate->getTensor().rows() << "x" << gate->getTensor().cols() << std::endl;
        root_->applyGate(gate->getTensor());
    }
}

void TTN::display() const {
    root_->display();
}
