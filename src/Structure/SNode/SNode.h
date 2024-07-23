#ifndef SNODE_H
#define SNODE_H

#include <string>
#include <vector>
#include <memory>

class SNode : public std::enable_shared_from_this<SNode> {
public:
    SNode(const std::string& name, std::shared_ptr<SNode> parent = nullptr, const std::vector<std::shared_ptr<SNode>>& children = {});

    void addChild(std::shared_ptr<SNode> child);
    const std::string& getName() const;
    std::shared_ptr<SNode> getParent() const;
    const std::vector<std::shared_ptr<SNode>>& getChildren() const;

    bool isLeaf() const;
    bool isRoot() const;
    void display(int depth = 0) const;

private:
    std::string name_;
    std::weak_ptr<SNode> parent_;
    std::vector<std::shared_ptr<SNode>> children_;
};

#endif // SNODE_H