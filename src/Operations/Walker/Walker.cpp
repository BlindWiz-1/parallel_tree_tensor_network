#include "Walker.h"

#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <queue>
#include <memory>
#include <vector>
#include "../../TTNCircuitSim/TNode/TNode.h"

std::vector<std::shared_ptr<TNode>> Walker::walk(const std::shared_ptr<TNode>& start, const std::shared_ptr<TNode>& stop) {
    std::vector<std::shared_ptr<TNode>> path;
    if (!start) return path;  // Return empty path if the start node is null

    // Debug: Print starting and stopping nodes
    std::cout << "Starting traversal from node: " << start->getName() << std::endl;
    if (stop) std::cout << "Stopping traversal at node: " << stop->getName() << std::endl;

    std::unordered_set<std::shared_ptr<TNode>> visited;
    std::function<void(const std::shared_ptr<TNode>&)> traverse;

    traverse = [&](const std::shared_ptr<TNode>& current) {
        if (!current || visited.count(current)) return;  // Avoid cycles or null nodes

        visited.insert(current);
        path.push_back(current);

        // If we reach the stop node, terminate the traversal
        if (current == stop) return;

        // Recur for all children nodes
        for (const auto& child : current->getChildren()) {
            traverse(child);
        }

        // If no stop node, also traverse upwards to the root
        if (!stop && current->getParent()) {
            traverse(current->getParent());
        }
    };

    traverse(start);

    // If there is a stop node, the path might need to be reversed.
    if (stop) {
        std::reverse(path.begin(), path.end());
    }

    return path;
}
