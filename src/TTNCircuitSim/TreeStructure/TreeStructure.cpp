//
// Created by Denado Rabeli on 7/31/24.
//
#include "TreeStructure.h"
#include <algorithm>

TreeStructure::TreeStructure(int clusters, int max_dim, int max_leaves, int sum_prod_dims)
    : clusters(clusters), max_dim(max_dim), max_leaves(max_leaves), sum_prod_dims(sum_prod_dims) {}

std::shared_ptr<SNode> findBestStructure(std::unordered_map<std::shared_ptr<SNode>, TreeStructure>& structures, const std::unordered_map<std::string, int>& arguments) {
    int bound = arguments.count("bound") ? arguments.at("bound") : 0;
    bool maximize_for_prod = arguments.count("maximize_for_prod") && arguments.at("maximize_for_prod");

    // Sort the structures by the desired properties
    std::vector<std::pair<std::shared_ptr<SNode>, TreeStructure>> sorted_structures(structures.begin(), structures.end());
    std::sort(sorted_structures.begin(), sorted_structures.end(), [maximize_for_prod](const auto& a, const auto& b) {
        int a_value = a.second.sum_prod_dims * (maximize_for_prod ? a.second.max_dim : 1);
        int b_value = b.second.sum_prod_dims * (maximize_for_prod ? b.second.max_dim : 1);
        if (a_value != b_value) return a_value < b_value;
        if (a.second.clusters * a.second.max_leaves != b.second.clusters * b.second.max_leaves) return a.second.clusters * a.second.max_leaves < b.second.clusters * b.second.max_leaves;
        return a.second.clusters > b.second.clusters;
    });

    // Filter out structures with big subtrees if the bound is set
    if (bound > 0) {
        sorted_structures.erase(std::remove_if(sorted_structures.begin(), sorted_structures.end(), [bound](const auto& s) {
            return s.second.max_dim >= (1 << bound);
        }), sorted_structures.end());
    }

    // Find the first structure that fulfills the desired property (max_dim < 2^max_leaves)
    auto it = std::find_if(sorted_structures.begin(), sorted_structures.end(), [](const auto& s) {
        return s.second.max_dim < (1 << s.second.max_leaves);
    });
    if (it != sorted_structures.end()) {
        return it->first;
    }

    // If no structure fulfills the desired property, return the first one
    return sorted_structures[0].first;
}
