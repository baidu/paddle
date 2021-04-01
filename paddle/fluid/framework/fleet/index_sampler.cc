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
                auto hierarchical_user = tree_->get_ancestor_given_level(user_inputs[i], max_layer - j);
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
            for (int idx_offset = 1; idx_offset <= layer_counts_[j]; idx_offset++) {
                int sample_res = 0;
                do {
                    sample_res = sampler_vec[j]->Sample();
                } while (layer_ids[j][sample_res] != ancestors[i][j]);
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


// ✧ 样本
//  ➢ User-Item-Label输入
//  ➢ User-Item-Label - Map出PathCodes
//  ➢ User-Item-Label=0 - 拼接随机采样出的PathCodes
//  ➢ 在EM 第一步之后 随机初始化Map函数 之后 汇合 输入NN；
//  ➢ 输入给NN的是 User-Item-Label-PathCodes；
//      ■ User作为输入，Label引导Beam Search去选取最大or最小概率； Item&PathCodes；优化使得PathCodes概率最大；

std::vector<std::vector<uint64_t>> GraphRandomSampler::sample1(std::vector<std::vector<uint64_t>>& user_inputs, std::vector<uint64_t>& target_ids, labels[i],int k1) {   //随机采样 输入多少样本的 多少倍 
    auto input_num = target_ids.size();
    auto user_feature_num = user_inputs[0].size();
    auto layer_nums = graph_->depth();
    auto node_nums = graph_->width();
    // auto beamSearch = new BeamSearchSampler("");
    auto itemtoPath = new itemToPaths("");  //
    auto pathstoItem = new pathsToItem("");  //
    // pathCodes=itemtoPath.pathCodes_
    // [x[0] for x in arr]

    pathCode_num = beamSearch.k_;

    std::vector<string> res_path(random_nums = （input_num*k1 + input_num） * pathCode_num, std::vector<uint64_t>(layer_nums));
    std::vector<Sampler*> sampler_vec(layer_nums);
    std::vector<std::vector<uint64_t>> outputs(train_nums=input_num*k1 + input_num, std::vector<uint64_t>(user_feature_num + 2 + pathCode_num));

    for (int cur_layer = 0; cur_layer <= layer_nums; cur_layer++){
        sampler_vec[cur_layer] = new paddle::operators::math::UniformSampler(node_nums, seed_);)
    }

    for (size_t ki = 0; ki < k1; ki++){
        sampler_vec[ki] = new paddle::operators::math::UniformSampler(input_num, seed_);)
    }

    idx = 0;    
    for (size_t i = 0; i < input_num; i++) {
        for (size_t ii = 0; ii < user_feature_num; ii++) {  //用户feature添加
            outputs[idx+i][ii] = user_inputs[i][ii];
        }  
        outputs[idx+i][user_feature_num] = target_ids[i];
        outputs[idx+i][user_feature_num+1] = labels[i];
        for (size_t jj = 0; jj < pathCode_num; jj++) {   //映射的多条初始化路径添加
            outputs[idx+i][user_feature_num+1+jj] = itemtoPath.pathCodes(target_ids[i])];  //改成hashMapList
        }
       // outputs[idx][ii+3] = itemtoPath.pathCodes(target_ids[i]); 

        // outputs.append([user_inputs[:][i], target_ids[i], labels[i], itemtoPath.pathCodes(target_ids[i])])  //
        for (size_t ki = 0; ki < k1; ki++){ //为每个User-Item-Label样本 扩展K1条item 该用户的负样本
            uint64_t cur_item_id = std::to_string(sampler_vec[j]->Sample());   //该用户的随机负样本映射
            for (size_t ii = 0; ii < user_feature_num; ii++) {  //用户feature添加
                outputs[idx+ki][ii] = user_inputs[i][ii];
            }  
            outputs[idx+ki][user_feature_num] = cur_item_id;
            outputs[idx+ki][user_feature_num+1] = 0;           
            for (size_t j = 0; j < pathCode_num; j++) { // 一个Item 映射到 pathCode_num条路径 List
                string cur_paths[j]=""
                for (int cur_layer = 0; cur_layer <= layer_nums; cur_layer++){
                    cur_paths[j]+=std::to_string(sampler_vec[j]->Sample());  //拼接一条随机出来的路径
                } 
                outputs[idx+ki][user_feature_num+1+j] = cur_paths[j];
            }  //pathCode_num  条路径 表征一个Item
            outputs.append([user_inputs[i], cur_item_id, labels[i], cur_paths]);
            idx += k1;
        }
    }
    return outputs;
}



// user-item-label-path  ==> pathCode   pi

// user-item-label-path  ==> item       pi  这个不需要  

// 既然是映射关系 那可以建立 itemEmbedding-pathCode的映射关系？？由用户点击ID 与ID的Embedding 先后联合训练？？
    // std::vector<std::vector<uint64_t>> outputs(train_nums=input_num*k1 + input_num, std::vector<uint64_t>(user_feature_num + 1 + item_embedding_num + pathCode_num));
    //map<k,[vals]>
    
// // GraphRandomSampler 生成随机路径编码 
// std::vector<std::vector<uint64_t>> GraphNegativeSampler::sample(std::vector<std::vector<uint64_t>>& user_inputs, std::vector<uint64_t>& target_ids, std::vector<bool> label) {
//     auto input_num = target_ids.size();
//     // auto neg_num = label.size()；/////
//     auto user_feature_num = user_inputs[0].size();
//     std::vector<std::vector<uint64_t>> outputs(train_nums=input_num*k1 + input_num, std::vector<uint64_t>(user_feature_num + 2 + pathCode_num));
//     //train_nums=input_num*k1 + input_num；           ；//neg_num*k2 + pos_num
//     auto layer_nums = graph_->depth();
//     auto node_nums = graph_->width();
//     auto item = map_->width();


//     std::vector<std::vector<uint64_t>> res_path(random_nums = input_nums * k1, std::vector<uint64_t>(layer_nums));
//     std::vector<Sampler*> sampler_vec(layer_nums);
//     for (int cur_layer = 0; cur_layer <= layer_nums; cur_layer++){
//         sampler_vec[cur_layer] = new paddle::operators::math::UniformSampler(node_nums, seed_);)
//     }
    
//     for (size_t i = 0; i < random_nums; i++) {
//         for (size_t j = 0; j < k; j++) {
//             for (int cur_layer = 0; cur_layer <= layer_nums; cur_layer++){
//                 res_path[i].append(sampler_vec[j]->Sample());
//             } 
//         }
//     }
//     return res_path;

//     // auto input_num = target_ids.size();
//     // auto neg_num = label.size()；/////
//     // auto user_feature_num = user_inputs[0].size();

//     // std::vector<std::vector<uint64_t>> outputs(train_nums=input_num*k1 + neg_num*k2 + (input_num-neg_num), std::vector<uint64_t>(user_feature_num + 2));
//     // //train_nums=input_num*k1 + neg_num*k2 + pos_num

// }



///



// std::vector<std::vector<uint64_t>> GraphRandomSampler::sample1(std::vector<std::vector<uint64_t>>& user_inputs, std::vector<uint64_t>& target_ids, std::vector<uint64_t>& labels,int k1) {   //随机采样 输入多少样本的 多少倍 
//     auto input_num = target_ids.size();
//     auto user_feature_num = user_inputs[0].size();
//     auto layer_nums = graph_->depth();
//     auto node_nums = graph_->width();
//     auto beamSearch = new BeamSearchSampler("");
//     auto itemtoPath = new itemToPaths("");  //
//     auto pathstoItem = new pathsToItem("");  //
//     // pathCodes=itemtoPath.pathCodes_
//     // [x[0] for x in arr]

//     pathCode_num = beamSearch.k_;

//     std::vector<string> res_path(random_nums = （input_num*k1 + input_num） * pathCode_num, std::vector<uint64_t>(layer_nums));
//     std::vector<Sampler*> sampler_vec(layer_nums);
//     std::vector<std::vector<uint64_t>> outputs(train_nums=input_num*k1 + input_num, std::vector<uint64_t>(user_feature_num + 2 + pathCode_num));

//     for (int cur_layer = 0; cur_layer <= layer_nums; cur_layer++){
//         sampler_vec[cur_layer] = new paddle::operators::math::UniformSampler(node_nums, seed_);)
//     }
    
//     for (size_t i = 0; i < input_num; i++) {
//         outputs.append([user_inputs[:][i], target_ids[i], labels[i], itemtoPath.pathCodes(target_ids[i])])  //
//         for (size_t ki = 0; ki < k1; ki++){ //为每个User-Item-Label样本 扩展K1条 该用户的负样本
            
//             for (size_t j = 0; j < pathCode_num; j++) { // 一个Item pathCode_num条路径 List
//                 cur_paths[j]=""
//                 for (int cur_layer = 0; cur_layer <= layer_nums; cur_layer++){
//                     cur_paths[j]+=std::to_string(sampler_vec[j]->Sample());  //拼接一条路径
//                 } 
//             }  //pathCode_num  条路径 表征一个Item

//             outputs.append([user_inputs[[i]], pathstoItem.getItem(cur_paths), labels[i], cur_paths])
//         }
//     }

//     return outputs;
// }



// // GraphRandomSampler 生成随机路径编码  //pathCode改成String拼接
// std::vector<std::vector<uint64_t>> GraphRandomSampler::sample(int input_nums, int k1) {   //随机采样 输入多少样本的 多少倍 
//     auto layer_nums = graph_->depth();
//     auto node_nums = graph_->width();

//     std::vector<std::vector<uint64_t>> res_path(random_nums = input_nums * k1, std::vector<uint64_t>(layer_nums));
//     std::vector<Sampler*> sampler_vec(layer_nums);
//     for (int cur_layer = 0; cur_layer <= layer_nums; cur_layer++){
//         sampler_vec[cur_layer] = new paddle::operators::math::UniformSampler(node_nums, seed_);)
//     }
    
//     for (size_t i = 0; i < random_nums; i++) {
//         for (size_t j = 0; j < k; j++) {
//             for (int cur_layer = 0; cur_layer <= layer_nums; cur_layer++){
//                 res_path[i].append(sampler_vec[j]->Sample());
//             } 
//         }
//     }
//     return res_path;

//     // auto input_num = target_ids.size();
//     // auto neg_num = label.size()；/////
//     // auto user_feature_num = user_inputs[0].size();

//     // std::vector<std::vector<uint64_t>> outputs(train_nums=input_num*k1 + neg_num*k2 + (input_num-neg_num), std::vector<uint64_t>(user_feature_num + 2));
//     // //train_nums=input_num*k1 + neg_num*k2 + pos_num

// }


// std::vector<std::vector<uint64_t>> outputs(train_nums=input_num*k1 + neg_num*k2 + (input_num-neg_num)*k3, std::vector<uint64_t>(user_feature_num + 2));

// std::vector<std::vector<uint64_t>> outputs(train_nums=input_num*k1 + input_num, std::vector<uint64_t>(user_feature_num + 2), pathCode);

// user-item-label-path  ==> pathCode   pi

// user-item-label-path  ==> item       pi


// std::vector<std::vector<uint64_t>> GraphNegativeSampler::sample(std::vector<std::vector<uint64_t>>& user_inputs, std::vector<uint64_t>& target_ids, std::vector<bool> label) {
//     auto input_num = target_ids.size();
//     auto neg_num = label.size()；/////
//     auto user_feature_num = user_inputs[0].size();


//     std::vector<std::vector<uint64_t>> outputs(train_nums=input_num*k1 + neg_num*k2 + (input_num-neg_num), std::vector<uint64_t>(user_feature_num + 2));
//     //train_nums=input_num*k1 + input_num；           ；//neg_num*k2 + pos_num
    
//     auto max_layer = tree_->height();
//     std::vector<Sampler*> sampler_vec(max_layer - start_sample_layer_);
//     std::vector<std::vector<uint64_t>> layer_ids(max_layer - start_sample_layer_);

//     auto layer_index = max_layer-1;
//     size_t idx = 0;
//     while (layer_index >= start_sample_layer_) {
//       layer_ids[idx] = tree_->get_nodes_given_level(layer_index);
//       sampler_vec[idx] = new paddle::operators::math::UniformSampler(layer_ids[idx].size(), seed_);
//       layer_index --;
//       idx ++;
//     }

//     auto ancestors = tree_->get_parent_path(target_ids, start_sample_layer_);
//     idx = 0;
//     for (size_t i = 0; i < input_num; i++) {
//         for (size_t j = 0; j < ancestors[i].size(); j++) {
//             // user
//             if (j > 0 && with_hierarchy) {
//                 auto hierarchical_user = tree_->get_ancestor_given_level(user_inputs[i], max_layer - j);
//                 for (int idx_offset = 0; idx_offset <= layer_counts_[j]; idx_offset++) {
//                     for (size_t k = 0; k < user_feature_num; k++) {
//                         outputs[idx+idx_offset][k] = hierarchical_user[k];
//                     }
//                 }
//             } else {
//                 for (int idx_offset = 0; idx_offset <= layer_counts_[j]; idx_offset++) {
//                     for (size_t k = 0; k < user_feature_num; k++) {
//                         outputs[idx + idx_offset][k] = user_inputs[i][k];
//                     }
//                 }
//             }
//             // sampler ++
//             outputs[idx][user_feature_num] = ancestors[i][j];
//             outputs[idx][user_feature_num + 1] = 1.0;
//             for (int idx_offset = 1; idx_offset <= layer_counts_[j]; idx_offset++) {
//                 int sample_res = 0;
//                 do {
//                     sample_res = sampler_vec[j]->Sample();
//                 } while (layer_ids[j][sample_res] == ancestors[i][j]);
//                 outputs[idx + idx_offset][user_feature_num] = layer_ids[j][sample_res];
//                 outputs[idx + idx_offset][user_feature_num + 1] = 0;
//             }
//             idx += layer_counts_[j];
//         }
//     }
//     return outputs;
// }


// std::vector<std::vector<uint64_t>> GraphRandomSampler::sample(std::vector<std::vector<uint64_t>>& user_inputs, std::vector<uint64_t>& target_ids, bool with_hierarchy) {
//     std::vector<std::vector<uint64_t>> outputs;
//     auto input_num = target_ids.size();  
//     auto user_feature_num = user_inputs[0].size();
//     std::vector<std::vector<uint64_t>> outputs(input_num * ()), std::vector<uint64_t>(user_feature_num + 2));
    
//     auto max_layer = graph_->height();
//     auto layer_nodeNums = graph_->branch();


//     std::vector<Sampler*> sampler_vec(max_layer - start_sample_layer_); 
//     std::vector<std::vector<uint64_t>> layer_ids(max_layer - start_sample_layer_);

//     auto layer_index = max_layer-1;
//     size_t idx = 0;
//     while (layer_index >= start_sample_layer_) {
//       layer_ids[idx] = graph_->get_nodes_given_level(layer_index);
//       sampler_vec[idx] = new paddle::operators::math::UniformSampler(layer_ids[idx].size(), seed_);
//       layer_index --;
//       idx ++;
//     }
//     return outputs;
// }


}  // end namespace framework
}  // end namespace paddle