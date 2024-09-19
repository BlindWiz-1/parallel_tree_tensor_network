//
// Created by Denado Rabeli on 7/19/24.
//
#include "Walker.h"
#include <unordered_set>
#include <algorithm>

std::vector<std::shared_ptr<TNode>> Walker::walk(const std::shared_ptr<TNode>& start, const std::shared_ptr<TNode>& stop) {
    std::vector<std::shared_ptr<TNode>> path;
    std::unordered_set<std::shared_ptr<TNode>> visited;

    std::function<void(const std::shared_ptr<TNode>&)> traverse = [&](const std::shared_ptr<TNode>& current) {
        if (!current || visited.count(current)) return;
        visited.insert(current);
        path.push_back(current);

        if (current == stop) return; // Stop the traversal if we reach the stop node

        for (const auto& child : current->getChildren()) {
            traverse(child);
        }

        if (current->getParent()) {
            traverse(current->getParent());
        }
    };

    traverse(start);
    return path;
}
