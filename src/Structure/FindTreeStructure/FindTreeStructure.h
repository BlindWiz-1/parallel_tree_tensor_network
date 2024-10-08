#ifndef FIND_TREE_STRUCTURE_H
#define FIND_TREE_STRUCTURE_H

#include "../../Circuits/Circuit/Circuit.h"
#include "../SNode/SNode.h"
#include <Eigen/Dense>
#include <memory>
#include <vector>

std::shared_ptr<SNode> findTreeStructure(const Circuit& circuit, int random_state = -1, bool flat = false);

Eigen::MatrixXd toSimilarityMatrix(const Circuit& circuit);

std::shared_ptr<SNode> createSubtree(const std::vector<int>& leaves, const Eigen::MatrixXd& similarity, int cluster);

int maxClusterSize(const std::vector<int>& labels);

std::shared_ptr<SNode> createSubtreeFlat(const std::vector<int>& leaves, int cluster);

#endif
