// Circuit.cpp
#include "Circuit.h"
#include <iostream>

Circuit::Circuit(int l_sites, int local_dimension)
    : l_sites_(l_sites), local_dimension_(local_dimension) {}

void Circuit::appendGate(const CircuitGate& gate) {
    for (const int site : gate.getSites()) {
        assert(0 <= site && site < l_sites_);
    }
    gates_.push_back(gate);
}

void Circuit::display() const {
    std::cout << "Circuit with " << l_sites_ << " sites and local dimension " << local_dimension_ << ":\n";
    for (const auto& gate : gates_) {
        gate.display();
    }
}
