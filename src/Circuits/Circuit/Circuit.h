#ifndef CIRCUIT_H
#define CIRCUIT_H

#include "../CircuitGate/CircuitGate.h"
#include <vector>

class Circuit {
public:
    Circuit(int l_sites, int local_dimension);

    void appendGate(const CircuitGate& gate);
    void display() const;

    int getLSites() const;
    const std::vector<CircuitGate>& getGates() const;

private:
    int l_sites_;
    int local_dimension_;
    std::vector<CircuitGate> gates_;
};

#endif // CIRCUIT_H
