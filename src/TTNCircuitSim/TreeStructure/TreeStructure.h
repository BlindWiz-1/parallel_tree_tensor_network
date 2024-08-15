//
// Created by Denado Rabeli on 7/31/24.
//
#ifndef TREE_STRUCTURE_H
#define TREE_STRUCTURE_H

#include <memory>
#include <unordered_map>
#include "../../Structure/SNode/SNode.h"

class TreeStructure {
public:
    int clusters;
    int max_dim;
    int max_leaves;
    int sum_prod_dims;

    TreeStructure(int clusters, int max_dim, int max_leaves, int sum_prod_dims);
};

std::shared_ptr<SNode> findBestStructure(std::unordered_map<std::shared_ptr<SNode>, TreeStructure>& structures, const std::unordered_map<std::string, int>& arguments);

#endif // TREE_STRUCTURE_H
