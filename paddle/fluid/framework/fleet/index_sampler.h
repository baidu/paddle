#include <vector>
#include "paddle/fluid/framework/fleet/index_wrapper.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

class IndexSampler {
 public:
  virtual ~IndexSampler() {}
  IndexSampler() {}

  template <typename T>
  static std::shared_ptr<IndexSampler> Init(const std::string& name) {
    std::shared_ptr<IndexSampler> instance = nullptr;
    instance.reset(new T(name));
    return instance;
  }

  virtual void init_layerwise_conf(const std::vector<int64_t> &layer_sample_counts, int start_sample_layer=1, int seed=0) {};
  virtual void init_beamsearch_conf(const int64_t k) {};
  virtual std::vector<std::vector<uint64_t>> sample(std::vector<std::vector<uint64_t>>& user_inputs, std::vector<uint64_t>& input_targets, bool with_hierarchy=false) = 0;
};

class LayerWiseSampler : public IndexSampler {
 public:
  virtual ~LayerWiseSampler() {}
  LayerWiseSampler(const std::string& name) {tree_ = IndexWrapper::GetInstance()->GetTreeIndex(name);}

  void init_layerwise_conf(const std::vector<int64_t> &layer_sample_counts, int start_sample_layer, int seed) override {
      seed_ = seed;
      start_sample_layer_ = start_sample_layer;

      PADDLE_ENFORCE_GT(start_sample_layer_, 0, "start sampler layer should greater than 0");
      PADDLE_ENFORCE_LT(start_sample_layer_, tree_->height(), "start sampler layer should less than max_layer");
    
      size_t i = 0;
      layer_counts_sum_ = 0;
      layer_counts_.clear();
      int cur_layer = start_sample_layer_;
      while (cur_layer < tree_->height()) {
          layer_counts_sum_ += layer_sample_counts[i] + 1;
          layer_counts_.push_back(layer_sample_counts[i]);
          VLOG(0) << "[INFO] level " << cur_layer << " sample_layer_counts.push_back: " << layer_sample_counts[i];
          cur_layer += 1;
          i += 1;
      }
      reverse(layer_counts_.begin(), layer_counts_.end());
      VLOG(0) << "sample counts sum: " << layer_counts_sum_;
  }
  std::vector<std::vector<uint64_t>> sample(std::vector<std::vector<uint64_t>>& user_inputs, std::vector<uint64_t>& target_ids, bool with_hierarchy) override;
 
 private:
  std::vector<int> layer_counts_;
  int64_t layer_counts_sum_{0};
  std::shared_ptr<TreeIndex> tree_{nullptr};
  int seed_{0};
  int start_sample_layer_{1};
};

class BeamSearchSampler : public IndexSampler{
 public:
  virtual ~BeamSearchSampler() {}
  BeamSearchSampler(const std::string& name) {tree_ = IndexWrapper::GetInstance()->GetTreeIndex(name);}

  void init_beamsearch_conf(const int64_t k) override {
    k_ = k;
    return;
  }
  std::vector<std::vector<uint64_t>> sample(std::vector<std::vector<uint64_t>>& user_inputs, std::vector<uint64_t>& target_ids, bool with_hierarchy) override;
 
 private:
  int64_t k_;
  std::shared_ptr<TreeIndex> tree_{nullptr};
};

}  // end namespace framework
}  // end namespace paddle