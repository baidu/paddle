#ifndef ECCL_TYPES
#define ECCL_TYPES

#include <string>

// version
using EcclVersion = struct EcclVersion_t {
    int major;
    int minor;
    int patch;

    std::string get_human_readable() const {
        return std::string(std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch));
    }
};

// unique id
const int ECCL_UNIQUE_ID_BYTES = 8192;
using EcclUniqueId = struct EcclUniqueId_t {
    char internal[ECCL_UNIQUE_ID_BYTES];
};

using EcclBackend = std::string;
using EcclScope = std::string;
using EcclRank = int;

// init configs
using EcclMemberRange = struct EcclMemberRange_t {
    EcclRank start;
    EcclRank end;
};

using EcclRankList = struct EcclRankList_t {
    EcclRank* ranks;
    int rank_count;
};

using EcclLayerMemberType = enum EcclLayerMemberType_t {
    UNKNOWN = 0,
    RANGE = 1,
    RANK_LIST = 2,
};

using EcclLayerConfig = struct EcclLayerConfig_t {
    union EcclLayerMembers_t {
        EcclMemberRange* range;
        EcclRankList* members;
    } members;
    EcclLayerMemberType member_type;
    EcclBackend backend;
    int level;
    EcclScope scope;
};

using EcclInitConfig = struct EcclInitConfig_t {
    EcclBackend backend;
    EcclScope scope;
    EcclLayerConfig* layers;
    int layer_count;
};

// group id
using PaddleEcclCommGroupIdType = std::string;
using EcclCommGroupIdType = const char*;
#define ECCL_COMM_GLOBAL_GROUP_ID  "__ECCL_COMM_GLOBAL_GROUP__"

// result
using EcclResult = enum EcclResult_t {
    SUCCESS = 0,
    INTERNAL_ERROR = 1,
};

// reduction operations
using EcclReductionOp = enum EcclReductionOp_t {
    SUM = 0,
    PROD = 1,
    MAX = 2,
    MIN = 3,
};

const int ECCL_REDUCTION_OP_COUNT = 4;

// Algorithms
using EcclCollectiveAlgorithm = enum EcclCollectiveAlgorithm_t {
    AUTO = 0,
    RING = 1,
    TREE = 2,
    HALVING_DOUBLE = 3,
    BCUBE = 4,
};

const int ECCL_COLLECTIVE_ALGORITHMS_COUNT = 5;

// data types
using EcclDataType = enum EcclDataType_t {
    DT_INT8 = 0, DT_CHAR = 0,
    DT_UINT8 = 1,
    DT_INT32 = 2, DT_INT=2,
    DT_UINT32 = 3,
    DT_INT64 = 4,
    DT_UINT64 = 5,
    DT_FLOAT16 = 6, DT_HALF = 6,
    DT_FLOAT32 = 7, DT_FLOAT = 7,
    DT_FLOAT64 = 8, DT_DOUBLE = 8,
};

const int ECCL_DATA_TYPE_COUNT = 9;

using EcclMemcpyType = enum EcclMemcpyType_t {
    ECCL_MEMCPY_HOST_TO_HOST,
    ECCL_MEMCPY_HOST_TO_DEVICE,
    ECCL_MEMCPY_DEVICE_TO_HOST,
    ECCL_MEMCPY_DEVICE_TO_DEVICE,
};


// device types
// using EcclDeviceType = std::string;
using EcclDeviceType = const char*;

// stream
using EcclRuntimeStream = void*;

#endif // ECCL_TYPES
