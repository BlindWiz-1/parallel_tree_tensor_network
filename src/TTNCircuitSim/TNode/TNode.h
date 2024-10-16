#ifndef TNODE_H
#define TNODE_H

#include <memory>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

using Tensor = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>;

class TNode : public std::enable_shared_from_this<TNode> {
public:
    TNode(const std::string& name, const Tensor& tensor, std::shared_ptr<TNode> parent = nullptr);

    void addChild(std::shared_ptr<TNode> child);
    const Tensor& getTensor() const;
    const std::string& getName() const;
    void setTensor(const Tensor& tensor);
    const std::unordered_map<std::string, int>& getLeafIndices() const;
    void setLeafIndices(const std::unordered_map<std::string, int>& leaf_indices);
    bool isLeaf() const;
    bool isRoot() const;
    void display(int depth = 0) const;
    const std::vector<std::shared_ptr<TNode>>& getChildren() const;
    std::shared_ptr<TNode> getParent() const;

    std::pair<double, int> countDimensions() const;

    void applyGate(const Tensor& gate_matrix);
    void applyGateAndReshape(const std::vector<Eigen::MatrixXcd>& update);

    std::shared_ptr<TNode> getItem(int key);
    std::vector<std::shared_ptr<TNode>> getItem(int site_i, int site_j);
    std::vector<std::shared_ptr<TNode>> getIntermediateNodes(int site_i, int site_j);

    std::shared_ptr<TNode> findRoot();

    int getTmpDim() const;
    void setTmpDim(int tmp_dim);

    int getTmpIndex() const;
    void setTmpIndex(int tmp_index);

    std::optional<Tensor> getTmpFactor() const;
    void setTmpFactor(const std::optional<Tensor>& tmp_factor);
    void update(int gate_dim, int site_i, int site_j);

private:
    Tensor contractFactorOnIndex(const Tensor& tensor, const Tensor& factor, int idx) const;

    std::string name_;
    Tensor tensor_;
    int local_dim_;
    std::shared_ptr<TNode> parent_;
    std::vector<std::shared_ptr<TNode>> children_;
    std::unordered_map<std::string, int> leaf_indices_;
    int tmp_dim_;
    int tmp_index_;
    std::optional<Tensor> tmp_factor_;
};

#endif // TNODE_H
