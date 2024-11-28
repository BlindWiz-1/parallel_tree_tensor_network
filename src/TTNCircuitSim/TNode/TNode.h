#ifndef TNODE_H
#define TNODE_H

#include <memory>
#include <vector>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using Tensor = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>;

class TNode : public std::enable_shared_from_this<TNode> {
public:
    TNode(const std::string& name, const Tensor& tensor, std::shared_ptr<TNode> parent = nullptr);
    TNode(const std::string& name, const Eigen::Tensor<std::complex<double>, 3>& tensor, std::shared_ptr<TNode> parent = nullptr);

    // Methods
    void addChild(std::shared_ptr<TNode> child);
    std::variant<Tensor, Eigen::Tensor<std::complex<double>, 3>> getTensor() const;
    void setTensor(const Tensor& tensor);  // For MatrixXcd
    void setTensor(const Eigen::Tensor<std::complex<double>, 3>& tensor);  // For Tensor type
    const std::string& getName() const;
    const std::unordered_map<std::string, int>& getLeafIndices() const;
    void setLeafIndices(const std::unordered_map<std::string, int>& leaf_indices);
    bool isLeaf() const;
    bool isRoot() const;
    const std::vector<std::shared_ptr<TNode>>& getChildren() const;
    int getTmpDim() const;
    void setTmpDim(int tmp_dim);
    int getTmpIndex() const;
    void setTmpIndex(int tmp_index);
    std::optional<Tensor> getTmpFactor() const;
    void setTmpFactor(const std::optional<Tensor>& tmp_factor);
    std::shared_ptr<TNode> findRoot();
    void display(int depth = 0) const;
    std::shared_ptr<TNode> getParent() const;
    std::pair<double, int> countDimensions() const;
    void applyGate(const Tensor& gate_matrix);
    void applyGateAndReshape(const std::vector<Tensor>& update);
    std::shared_ptr<TNode> getItem(const std::string& key);
    std::vector<std::shared_ptr<TNode>> getItem(const std::string& site_i, const std::string& site_j);
    std::vector<std::shared_ptr<TNode>> getPathToRoot(const std::string& site);
    std::vector<std::shared_ptr<TNode>> getIntermediateNodes(const std::string& site_i, const std::string& site_j);
    void update(int gate_dim, int site_i, int site_j);
    std::vector<int> getTensorDimensions() const;  // New method

private:
    std::string name_;
    std::optional<Tensor> tensor_matrix_;  // For leaf nodes
    std::optional<Eigen::Tensor<std::complex<double>, 3>> tensor_tensor_;  // For intermediate nodes
    int local_dim_;
    std::shared_ptr<TNode> parent_;
    std::vector<std::shared_ptr<TNode>> children_;
    std::unordered_map<std::string, int> leaf_indices_;
    int tmp_dim_;
    int tmp_index_;
    std::optional<Tensor> tmp_factor_;
};

#endif // TNODE_H
