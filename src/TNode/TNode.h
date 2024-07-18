#ifndef TNODE_H
#define TNODE_H

#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <iostream>

using Tensor = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>;

class TNode : public std::enable_shared_from_this<TNode> {
public:
    TNode(const std::string& name, const Tensor& tensor, std::shared_ptr<TNode> parent = nullptr);

    void addChild(std::shared_ptr<TNode> child);
    const Tensor& getTensor() const;
    const std::string& getName() const;
    const std::unordered_map<std::string, int>& getLeafIndices() const;
    void setLeafIndices(const std::unordered_map<std::string, int>& leaf_indices);
    bool isLeaf() const;
    bool isRoot() const;
    void display(int depth = 0) const;
    const std::vector<std::shared_ptr<TNode>>& getChildren() const;
    std::pair<double, int> countDimensions() const;

private:
    std::string name_;
    Tensor tensor_;
    int local_dim_;
    std::shared_ptr<TNode> parent_;
    std::vector<std::shared_ptr<TNode>> children_;
    std::unordered_map<std::string, int> leaf_indices_;
    int tmp_dim_;
    int tmp_index_;
};

#endif // TNODE_H
