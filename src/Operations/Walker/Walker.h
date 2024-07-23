//
// Created by Denado Rabeli on 7/19/24.
//

#ifndef WALKER_H
#define WALKER_H

#include <vector>
#include <memory>
#include "../../TTNCircuitSim/TNode/TNode.h"
#include "../../TTNCircuitSim/PseudoTNode/PseudoTNode.h"

class Walker {
public:
    std::vector<std::shared_ptr<TNode>> walk(const TNode* start, const TNode* stop);
    std::vector<std::shared_ptr<PseudoTNode>> walk(const PseudoTNode* start, const PseudoTNode* stop);
};

#endif // WALKER_H
