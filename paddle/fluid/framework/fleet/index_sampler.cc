#include "paddle/fluid/framework/fleet/index_sampler.h"
#include "paddle/fluid/operators/math/sampler.h"

namespace paddle {
namespace framework {

using Sampler = paddle::operators::math::Sampler;

std::vector<std::vector<uint64_t>> LayerWiseSampler::sample(std::vector<std::vector<uint64_t>>& user_inputs, std::vector<uint64_t>& target_ids, bool with_hierarchy) {
    auto input_num = target_ids.size();
    auto user_feature_num = user_inputs[0].size() ;
    std::vector<std::vector<uint64_t>> outputs(input_num * layer_counts_sum_, std::vector<uint64_t>(user_feature_num + 2));
    
    auto max_layer = tree_->height();
    std::vector<Sampler*> sampler_vec(max_layer - start_sample_layer_);
    std::vector<std::vector<uint64_t>> layer_ids(max_layer - start_sample_layer_);

    auto layer_index = max_layer-1;
    size_t idx = 0;
    while (layer_index >= start_sample_layer_) {
      layer_ids[idx] = tree_->get_nodes_given_level(layer_index);
      sampler_vec[idx] = new paddle::operators::math::UniformSampler(layer_ids[idx].size(), seed_);
      layer_index --;
      idx ++;
    }

    auto ancestors = tree_->get_parent_path(target_ids, start_sample_layer_);
    idx = 0;
    for (size_t i = 0; i < input_num; i++) {
        for (size_t j = 0; j < ancestors[i].size(); j++) {
            // user
            if (j > 0 && with_hierarchy) {
                auto hierarchical_user = tree_->get_ancestor_given_level(user_inputs[i], max_layer - j - 1);
                for (int idx_offset = 0; idx_offset <= layer_counts_[j]; idx_offset++) {
                    for (size_t k = 0; k < user_feature_num; k++) {
                        outputs[idx+idx_offset][k] = hierarchical_user[k];
                    }
                }
            } else {
                for (int idx_offset = 0; idx_offset <= layer_counts_[j]; idx_offset++) {
                    for (size_t k = 0; k < user_feature_num; k++) {
                        outputs[idx + idx_offset][k] = user_inputs[i][k];
                    }
                }
            }
            
            // sampler ++
            outputs[idx][user_feature_num] = ancestors[i][j];
            outputs[idx][user_feature_num + 1] = 1.0;
            idx += 1;
            for (int idx_offset = 0; idx_offset < layer_counts_[j]; idx_offset++) {
                int sample_res = 0;
                do {
                    sample_res = sampler_vec[j]->Sample();
                } while (layer_ids[j][sample_res] == ancestors[i][j]);
                VLOG(1) << sample_res << " " << layer_ids[j][sample_res] << " " << ancestors[i][j];
                outputs[idx + idx_offset][user_feature_num] = layer_ids[j][sample_res];
                outputs[idx + idx_offset][user_feature_num + 1] = 0;
            }
            idx += layer_counts_[j];
        }
    }
    return outputs;
}

std::vector<std::vector<uint64_t>> BeamSearchSampler::sample(std::vector<std::vector<uint64_t>>& user_inputs, std::vector<uint64_t>& target_ids, bool with_hierarchy) {
    std::vector<std::vector<uint64_t>> outputs;
    return outputs;
}

}  // end namespace framework
}  // end namespace paddle