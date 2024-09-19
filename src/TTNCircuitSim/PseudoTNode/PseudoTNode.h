#ifndef PSEUDOTNODE_H
#define PSEUDOTNODE_H

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <iostream>
#include <optional>
#include <Eigen/Dense>

using Tensor = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>;

class PseudoTNode : public std::enable_shared_from_this<PseudoTNode> {
public:
    PseudoTNode(const std::string& name, const std::vector<int>& shape, std::shared_ptr<PseudoTNode> parent = nullptr);

    void addChild(std::shared_ptr<PseudoTNode> child);
    const std::vector<int>& getShape() const;
    void setShape(const std::vector<int>& new_shape);
    const std::string& getName() const;
    const std::unordered_map<std::string, int>& getLeafIndices() const;
    void setLeafIndices(const std::unordered_map<std::string, int>& leaf_indices);
    bool isLeaf() const;
    bool isRoot() const;
    void display(int depth = 0) const;
    std::shared_ptr<PseudoTNode> getParent() const;
    std::vector<std::shared_ptr<PseudoTNode>> getChildren() const;

    void applyGate(const Tensor& gate_matrix);
    void applyGateAndReshape(const Tensor& update);
    // void update(int gate_dim, int site_i, int site_j);

private:
    std::string name_;
    std::vector<int> shape_;
    int local_dim_;
    std::shared_ptr<PseudoTNode> parent_;
    std::vector<std::shared_ptr<PseudoTNode>> children_;
    std::unordered_map<std::string, int> leaf_indices_;
};

#endif // PSEUDOTNODE_H
