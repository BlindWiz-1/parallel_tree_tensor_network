//
// Created by Denado Rabeli on 7/19/24.
//
#include "Walker.h"
#include <unordered_set>
#include <algorithm>

std::vector<std::shared_ptr<TNode>> Walker::walk(const TNode* start, const TNode* stop) {
    std::unordered_set<const TNode*> visited;
    std::vector<std::shared_ptr<TNode>> path;

    std::function<bool(const TNode*)> findPath = [&](const TNode* current) -> bool {
        if (!current || visited.count(current)) {
            return false;
        }
        visited.insert(current);
        path.push_back(const_cast<TNode*>(current)->shared_from_this());
        if (current == stop) {
            return true;
        }
        for (const auto& child : current->getChildren()) {
            if (findPath(child.get())) {
                return true;
            }
        }
        if (findPath(current->getParent().get())) {
            return true;
        }
        path.pop_back();
        return false;
    };

    if (findPath(start)) {
        std::reverse(path.begin(), path.end());
    } else {
        path.clear(); // Clear path if no path found
    }
    return path;
}

std::vector<std::shared_ptr<PseudoTNode>> Walker::walk(const PseudoTNode* start, const PseudoTNode* stop) {
    std::unordered_set<const PseudoTNode*> visited;
    std::vector<std::shared_ptr<PseudoTNode>> path;

    std::function<bool(const PseudoTNode*)> findPath = [&](const PseudoTNode* current) -> bool {
        if (!current || visited.count(current)) {
            return false;
        }
        visited.insert(current);
        path.push_back(const_cast<PseudoTNode*>(current)->shared_from_this());
        if (current == stop) {
            return true;
        }
        for (const auto& child : current->getChildren()) {
            if (findPath(child.get())) {
                return true;
            }
        }
        if (findPath(current->getParent().get())) {
            return true;
        }
        path.pop_back();
        return false;
    };

    if (findPath(start)) {
        std::reverse(path.begin(), path.end());
    } else {
        path.clear(); // Clear path if no path found
    }
    return path;
}
