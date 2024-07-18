#ifndef PSEUDOTNODE_H
#define PSEUDOTNODE_H

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>

class PseudoTNode : public std::enable_shared_from_this<PseudoTNode> {
public:
    PseudoTNode(const std::string& name, const std::vector<int>& shape, std::shared_ptr<PseudoTNode> parent = nullptr);

    void addChild(std::shared_ptr<PseudoTNode> child);
    const std::string& getName() const;
    const std::vector<int>& getShape() const;
    const std::unordered_map<std::string, int>& getLeafIndices() const;
    void setLeafIndices(const std::unordered_map<std::string, int>& leaf_indices); // Setter method

    bool isLeaf() const;
    bool isRoot() const;
    void display(int depth = 0) const;

private:
    std::string name_;
    std::vector<int> shape_;
    std::shared_ptr<PseudoTNode> parent_;
    std::vector<std::shared_ptr<PseudoTNode>> children_;
    std::unordered_map<std::string, int> leaf_indices_;
};

#endif // PSEUDOTNODE_H
