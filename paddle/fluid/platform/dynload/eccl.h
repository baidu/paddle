#ifndef ECCL_H
#define ECCL_H

#include <string>
#include <vector>

#include "eccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

//--------------------- Version ---------------------
// Get current ECCL version
EcclResult eccl_get_version(EcclVersion* version) noexcept;
//------------------ End of Version -----------------


//--------------------- Communication Management ---------------------
// Init
// EcclResult eccl_init(const EcclInitConfig* init_config);

// Generate ECCL unique id
EcclResult eccl_gen_unique_id(const int my_rank, const char* bootstrap_endpoint, const int rank_count, const int split_index,
    EcclCommGroupIdType comm_group_id);

// Initialize ECCL for rank
EcclResult eccl_init_comm_global(const int rank_count, const int my_rank,
    EcclDeviceType my_device_type, int my_device_id, EcclCommGroupIdType comm_group_id);

// Destroy the ECCL world communicator
EcclResult eccl_destroy_comm_global(EcclCommGroupIdType comm_group_id);

// Create communication sub-group
// EcclResult eccl_create_sub_comm_group(EcclCommGroupIdType comm_group_id, const int my_rank, const std::vector<int>& global_rank_ids);

// Destroy communication sub-group
// EcclResult eccl_destroy_sub_comm_group(EcclCommGroupIdType comm_group_id, const int my_rank);
//------------------ End of Communication Management -----------------



// //--------------------- High-level Collective Functions ---------------------
// // High-level collective functions provides easy-to-use APIs to end users.
// // These APIs takes minimal set of input params, all the other behaviours will
// // be default values.
// // For example, all high-level operations share the same device stream.
// // Also, the collective algorithms will be picked by ECCL internal implementation.
// // These high level APIs are in async mode by default, these OPs can be synced with
// // eccl_wait OP.
// //---------------------------------------------------------------------------
// // Reduce
// EcclResult eccl_reduce(const void* sendbuff, void* recvbuff, size_t count, EcclDataType data_type,
//     EcclReductionOp op, int root, EcclCommGroupIdType group_id = ECCL_COMM_GLOBAL_GROUP_ID);

// // Broadcast
// EcclResult eccl_broadcast(const void* sendbuff, void* recvbuff, size_t count, EcclDataType data_type,
//     int root, EcclCommGroupIdType group_id = ECCL_COMM_GLOBAL_GROUP_ID);

// // All-Reduce
// EcclResult eccl_all_reduce(const void* sendbuff, void* recvbuff, size_t count, EcclDataType data_type,
//     EcclReductionOp op, EcclCommGroupIdType group_id = ECCL_COMM_GLOBAL_GROUP_ID);

// // Reduce-Scatter
// EcclResult eccl_reduce_scatter(const void* sendbuff, void* recvbuff, size_t recv_count, EcclDataType data_type,
//     EcclReductionOp op, EcclCommGroupIdType group_id = ECCL_COMM_GLOBAL_GROUP_ID);

// // All-Gather
// EcclResult eccl_all_gather(const void* sendbuff, void* recvbuff, size_t send_count, EcclDataType data_type,
//     EcclCommGroupIdType group_id = ECCL_COMM_GLOBAL_GROUP_ID);

// // Send
// EcclResult eccl_send(const void* sendbuff, size_t count, EcclDataType data_type, int peer,
//     EcclCommGroupIdType group_id = ECCL_COMM_GLOBAL_GROUP_ID);

// // Receive
// EcclResult eccl_recv(void* recvbuff, size_t count, EcclDataType data_type, int peer,
//     EcclCommGroupIdType group_id = ECCL_COMM_GLOBAL_GROUP_ID);

// // Synchronize
// EcclResult eccl_wait(EcclCommGroupIdType group_id = ECCL_COMM_GLOBAL_GROUP_ID);

// // Barrier
// EcclResult eccl_barrier(EcclCommGroupIdType group_id = ECCL_COMM_GLOBAL_GROUP_ID);
// //------------------ End of High-level Collective Functions -----------------


//--------------------- Fine-grained Collective Functions ---------------------
// Fine-grained collective functions allows users to control the operation
// stream, as well as the collective algorithms.
//-----------------------------------------------------------------------------
// Reduce
EcclResult eccl_reduce(const void* sendbuff, void* recvbuff, size_t count, EcclDataType data_type,
    EcclReductionOp op, int root, EcclCommGroupIdType group_id, EcclRuntimeStream stream, EcclCollectiveAlgorithm algo=AUTO);

// Broadcast
EcclResult eccl_broadcast(const void* sendbuff, void* recvbuff, size_t count, EcclDataType data_type,
    int root, EcclCommGroupIdType group_id, EcclRuntimeStream stream, EcclCollectiveAlgorithm algo=AUTO);

// All-Reduce
EcclResult eccl_all_reduce(const void* sendbuff, void* recvbuff, size_t count, EcclDataType data_type,
    EcclReductionOp op, EcclCommGroupIdType group_id, EcclRuntimeStream stream, EcclCollectiveAlgorithm algo=AUTO);

// Reduce-Scatter
EcclResult eccl_reduce_scatter(const void* sendbuff, void* recvbuff, size_t recv_count, EcclDataType data_type,
    EcclReductionOp op, EcclCommGroupIdType group_id, EcclRuntimeStream stream, EcclCollectiveAlgorithm algo=AUTO);

// All-Gather
EcclResult eccl_all_gather(const void* sendbuff, void* recvbuff, size_t send_count, EcclDataType data_type,
    EcclCommGroupIdType group_id, EcclRuntimeStream stream, EcclCollectiveAlgorithm algo=AUTO);

// Send
EcclResult eccl_send(const void* sendbuff, size_t count, EcclDataType data_type, int peer,
    EcclCommGroupIdType group_id, EcclRuntimeStream stream, EcclCollectiveAlgorithm algo=AUTO);

// Receive
EcclResult eccl_recv(void* recvbuff, size_t count, EcclDataType data_type, int peer,
    EcclCommGroupIdType group_id, EcclRuntimeStream stream, EcclCollectiveAlgorithm algo=AUTO);

// Barrier
EcclResult eccl_barrier(EcclCommGroupIdType group_id, EcclRuntimeStream stream);

// Sync stream
EcclResult eccl_sync_stream(EcclCommGroupIdType group_id, EcclRuntimeStream stream);
//------------------ End of Fine-grained Collective Functions -----------------


//--------------------- Group Semantics ---------------------
// Mark begin of a group
EcclResult eccl_group_begin();

// Mark end of a group, and actually run the OPs
EcclResult eccl_group_end();
//------------------ End of Group Semantics -----------------

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // ECCL_H
