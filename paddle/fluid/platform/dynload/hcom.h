/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file hcom.h
 * @brief HCOM API
 */

#ifndef HCOM_H_
#define HCOM_H_

// #include <runtime/rt.h>
#include "paddle/fluid/platform/dynload/hcom_type.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief Initialize HCOM.
 *
 * @param rank_table A string identifying the rank table file path, include file name.
 * @param identify A string identifying the identify for the rank.
 * @return hcclResult_t
 * @see hcom_destroy()
 */
extern hcclResult_t hcom_init(const char *rank_table, const char *identify);

/**
 * @brief Destroy HCOM
 *
 * @return hcclResult_t
 * @see hcom_init()
 */
extern hcclResult_t hcom_destroy(void);

/**
 * @brief Bind the model.
 *
 * @param model A pointer identifying the model information.
 * @param stream A pointer identifying the stream information.
 * @return hcclResult_t
 * @see hcom_unbind_model()
 */
extern hcclResult_t hcom_bind_model(rtModel_t model, rtStream_t stream);

/**
 * @brief Unbind the model.
 *
 * @param model An pointer identifying the model information.
 * @return hcclResult_t
 * @see hcom_unbind_model()
 */
extern hcclResult_t hcom_unbind_model(rtModel_t model);

/**
 * @brief All-gather operator.
 *
 * @param tag A string identifying the tag of the operator.
 * @param inputPtr A pointer identifying the input data address of the operator.
 * @param outputPtr A pointer identifying the output data address of the operator.
 * @param inputCount An integer(u64) identifying the number of the input data.
 * @param dataType The data type of the operator, must be one of the following types: int8, int32, float16, float32.
 * @param group A string identifying the group name of ranks participating in the operator.
 * @param stream A pointer identifying the stream information.
 * @return hcclResult_t 
 */
extern hcclResult_t hcom_all_gather(const char *tag, void *inputPtr, void *outputPtr, u64 inputCount,
                                    hcclDataType_t dataType, const char *group, rtStream_t stream);

/**
 * @brief All-reduce operator.
 *
 * @param tag A string identifying the tag of the operator.
 * @param inputPtr A pointer identifying the input data address of the operator.
 * @param outputPtr A pointer identifying the output data address of the operator.
 * @param count An integer(u64) identifying the number of the output data.
 * @param dataType The data type of the operator, must be one of the following types: int8, int32, float16, float32.
 * @param op The reduction type of the operator, must be one of the following types: sum, min, max, prod.
 * @param group A string identifying the group name of ranks participating in the operator.
 * @param stream A pointer identifying the stream information.
 * @return hcclResult_t 
 */
extern hcclResult_t hcom_all_reduce(const char *tag, void *inputPtr, void *outputPtr, u64 count,
                                    hcclDataType_t dataType, hcclRedOp_t op, const char *group, rtStream_t stream);

/**
 * @brief Broadcast operator.
 *
 * @param tag A string identifying the tag of the operator.
 * @param ptr A pointer identifying the data address of the operator.
 * @param count An integer(u64) identifying the number of the data.
 * @param dataType The data type of the operator, must be one of the following types: int8, int32, float16, float32.
 * @param root An integer(u32) identifying the the root rank in the operator.
 * @param group A string identifying the group name of ranks participating in the operator.
 * @param stream A pointer identifying the stream information.
 * @return hcclResult_t 
 */
extern hcclResult_t hcom_broadcast(const char *tag, void *ptr, u64 count, hcclDataType_t dataType, u32 root,
                                   const char *group, rtStream_t stream);

/**
 * @brief Reduce-scatter operator.
 *
 * @param tag A string identifying the tag of the operator.
 * @param inputPtr A pointer identifying the input data address of the operator.
 * @param outputPtr A pointer identifying the output data address of the operator.
 * @param count An integer(u64) identifying the number of the data.
 * @param dataType The data type of the operator, must be one of the following types: int8, int32, float16, float32.
 * @param op The reduction type of the operator, must be one of the following types: sum, min, max, prod.
 * @param group A string identifying the group name of ranks participating in the operator.
 * @param stream A pointer identifying the stream information.
 * @return hcclResult_t 
 */
extern hcclResult_t hcom_reduce_scatter(const char *tag, void *inputPtr, void *outputPtr, u64 count,
                                        hcclDataType_t dataType, hcclRedOp_t op, const char *group, rtStream_t stream);

/**
 * @brief Get the rank number in the group.
 *
 * @param group A string identifying the group name.
 * @param rankSize A pointer identifying the rank number.
 * @return hcclResult_t 
 */
hcclResult_t hcom_get_rank_size(const char *group, u32 *rankSize);

/**
 * @brief Get the rank number of this rank's server within the group.
 *
 * @param group A string identifying the group name.
 * @param localRankSize A pointer identifying the rank number.
 * @return hcclResult_t 
 */
hcclResult_t hcom_get_local_rank_size(const char *group, u32 *localRankSize);

/**
 * @brief Get the rank id of this rank.
 *
 * @param group A string identifying the group name.
 * @param rankId A pointer identifying the rank id.
 * @return hcclResult_t 
 */
hcclResult_t hcom_get_rank_id(const char *group, u32 *rankId);

/**
 * @brief Get the local rank id of this rank's server within the group.
 *
 * @param group A string identifying the group name.
 * @param localRankId A pointer identifying the local rank id.
 * @return hcclResult_t 
 */
hcclResult_t hcom_get_local_rank_id(const char *group, u32 *localRankId);

/**
 * @brief Get the world rank id according to the group rank id.
 *
 * @param group A string identifying the group name.
 * @param groupRank An integer(u32) identifying the group rank id.
 * @param worldRank A pointer identifying the world rank id.
 * @return hcclResult_t 
 */
hcclResult_t hcom_get_world_rank_from_group_rank(const char *group, u32 groupRank, u32 *worldRank);

/**
 * @brief Get the group rank id according to the world rank id.
 *
 * @param worldRank An integer(u32) identifying the world rank id.
 * @param group A string identifying the group name.
 * @param groupRank A pointer identifying the group rank id.
 * @return hcclResult_t 
 */
hcclResult_t hcom_get_group_rank_from_world_rank(u32 worldRank, const char *group, u32 *groupRank);

/**
 * @brief Create group.
 *
 * @param group A string identifying the group name.
 * @param rankNum An integer(u32) identifying the number of ranks in the group.
 * @param rankIds A list identifying the ranks in the group.
 * @return hcclResult_t 
 */
hcclResult_t hcom_create_group(const char *group, u32 rankNum, u32 *rankIds);

/**
 * @brief Destroy group
 *
 * @param group A string identifying the group name.
 * @return hcclResult_t 
 */
hcclResult_t hcom_destroy_group(const char *group);

/**
 * @brief Send operator.
 *
 * @param tag A string identifying the tag of the operator.
 * @param inputPtr A pointer identifying the input data address of the operator.
 * @param count An integer(u64) identifying the number of the data.
 * @param dataType The data type of the operator, must be one of the following types: int8, int32, float16, float32.
 * @param destRank An integer identifying the destination rank.
 * @param srTag An integer identifying the send/recv message tag.
 * The message will be send by the receive operator with the same "sr_tag".
 * @param group A string identifying the group name of ranks participating in the operator.
 * @param stream A pointer identifying the stream information.
 * @return hcclResult_t 
 */
hcclResult_t hcom_send(const char *tag, void *inputPtr, u64 count, hcclDataType_t dataType,
    u32 destRank, u32 srTag, const char *group, rtStream_t stream);

/**
 * @brief Receive operator.
 *
 * @param tag A string identifying the tag of the operator.
 * @param outputPtr A pointer identifying the output data address of the operator.
 * @param count An integer(u64) identifying the number of the data.
 * @param dataType The data type of the operator, must be one of the following types: int8, int32, float16, float32.
 * @param srcRank An integer identifying the source rank.
 * @param srTag An integer identifying the send/recv message tag. 
 * The message will be send by the send operator with the same "sr_tag".
 * @param group A string identifying the group name of ranks participating in the operator.
 * @param stream A pointer identifying the stream information.
 * @return hcclResult_t 
 */
hcclResult_t hcom_receive(const char *tag, void *outputPtr, u64 count, hcclDataType_t dataType,
    u32 srcRank, u32 srTag, const char *group, rtStream_t stream);

/**
 * @brief Get the gradient split strategy with in the group.
 *
 * @param group A string identifying the group name.
 * @param feature A pointer identifying the feature of the model.
 * @param maxSegmentNum An integer(u32) identifying the max segments of gradients.
 * @param segmentNum A pointer identifying the segments number of gradients.
 * @param segmentIdx A list identifying the index of end gradient in each segment.
 * @return hcclResult_t 
 */
hcclResult_t hcom_get_split_strategy(const char *group, const struct model_feature *feature, 
    u32 maxSegmentNum, u32 *segmentNum, u32 *segmentIdx, GradSplitForceMode force = FORCE_NONE);

/**
 * @brief Set the gradient split strategy with in the group, according to gradient index.
 *
 * @param group A string identifying the group name.
 * @param segmentNum An integer(u32) identifying the segments number of gradients.
 * @param IdxList A list identifying the index of end gradient in each segment.
 * @return hcclResult_t
 */
extern hcclResult_t hcom_set_split_strategy_by_index(const char *group, u32 segmentNum, const u32 *IdxList);

/**
 * @brief Set the gradient split strategy with in the group, according to gradient data size.
 *
 * @param group A string identifying the group name.
 * @param segmentNum An integer(u32) identifying the segments number of gradients.
 * @param sizeList A list identifying the percent of each segment.
 * @return hcclResult_t
 */
extern hcclResult_t hcom_set_split_strategy_by_size(const char *group, u32 segmentNum, const float *sizeList);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCOM_H_
