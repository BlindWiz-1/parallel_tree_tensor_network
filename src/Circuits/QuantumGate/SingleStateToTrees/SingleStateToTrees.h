//
// Created by Denado Rabeli on 7/18/24.
//

#ifndef SINGLESTATETOTREES_H
#define SINGLESTATETOTREES_H

#include "../SNode/SNode.h"
#include "../TNode/TNode.h"
#include "../PseudoTNode/PseudoTNode.h"
#include <vector>
#include <vector>
#include <memory>
#include <string>

std::shared_ptr<TNode> singleStatesToTree(const std::vector<int>& single_states, int local_dimension, const std::shared_ptr<SNode>& structure);
std::shared_ptr<PseudoTNode> createPseudoTree(const std::shared_ptr<SNode>& node, const std::vector<int>& single_states, int local_dimension);

std::shared_ptr<TNode> createTreeLevel(const std::shared_ptr<SNode>& node, const std::vector<int>& single_states, int local_dimension);
std::shared_ptr<PseudoTNode> createPseudoTreeLevel(const std::shared_ptr<SNode>& node, const std::vector<int>& single_states, int local_dimension);

#endif
