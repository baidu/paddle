#ifndef ECCL_UTILS_H
#define ECCL_UTILS_H
#include "eccl_types.h"

#include <acl/acl.h>
#include <acl/acl_base.h>

#include <hccl/hccl.h>
#include <hccl/hccl_types.h>

// for hccl adapter
constexpr auto kHcclConfigFile = "HCCL_CONFIG_PATH";

#define ACL_SUCCESS 0
#define ACLCHECK(cmd) do {                         \
  aclError e = cmd;                              \
  if( e != ACL_SUCCESS) {                          \
    printf("Failed: acl error %s:%d '%d'\n",             \
        __FILE__,__LINE__,e);   \
    exit(-1);                             \
  }                                                 \
} while(0)

#define HCCLCHECK(cmd) do {                         \
  HcclResult r = cmd;                             \
  if (r!= HCCL_SUCCESS) {                            \
    printf("Failed, HCCL error %s:%d '%d'\n",             \
        __FILE__,__LINE__,r);   \
    exit(-1);                             \
  }                                                 \
} while(0)

// for common utils eccl
#define ECCL_ENSURE_SUCCESS(call) do { \
  EcclResult res = call; \
  if (res != SUCCESS) { \
    /* Print the back trace*/ \
    return res; \
  } \
} while (0);

#endif //ECCL_UTILS_H
