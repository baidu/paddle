/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/gen_hccl_id_op_helper.h"
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <sys/socket.h>

#include <algorithm>
#include <ostream>
#include <string>

#include "glog/logging.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/split.h"

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace paddle {
namespace operators {

constexpr char COMM_HEAD[] = "_pd_gen_comm_id_";
#define HCCL_UNIQUE_ID_BYTES 1024

// Check system calls, such as socket, bind.
#define CHECK_SYS_CALL(call, name)          \
  do {                                      \
    int retval;                             \
    CHECK_SYS_CALL_VAL(call, name, retval); \
  } while (false)

#define CHECK_SYS_CALL_VAL(call, name, retval)                            \
  do {                                                                    \
    RETRY_SYS_CALL_VAL(call, name, retval);                               \
    if (retval == -1) {                                                   \
      PADDLE_THROW(platform::errors::Unavailable("Call to %s failed: %s", \
                                                 name, strerror(errno))); \
    }                                                                     \
  } while (false)

#define RETRY_SYS_CALL_VAL(call, name, retval)                           \
  do {                                                                   \
    retval = (call);                                                     \
    if (retval == -1 &&                                                  \
        (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) {   \
      LOG(WARNING) << "Call " << name << " returned " << strerror(errno) \
                   << " retry";                                          \
    } else {                                                             \
      break;                                                             \
    }                                                                    \
  } while (true)

static int SocketSend(int fd, const char* buffer, int size) {
  int offset = 0;
  int bytes = 0;
  while (offset < size) {
    bytes = send(fd, buffer + offset, size - offset, 0);
    if (bytes == -1) {
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
        // send failed
        return -1;
      } else {
        bytes = 0;
      }
    }
    offset += bytes;
  }
  return offset;
}

static int SocketRecv(int fd, char* buffer, int size) {
  int offset = 0;
  int bytes = 0;
  while (offset < size) {
    bytes = recv(fd, buffer + offset, size - offset, 0);
    if (bytes == 0) {
      // closed by client, maybe probing alive client
      return 0;
    }
    if (bytes == -1) {
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
        return -1;
      } else {
        bytes = 0;
      }
    }
    offset += bytes;
  }
  return offset;
}

static void BindOrConnectFailed(int timeout, int* try_times, int* total_time,
                                const char* op, const std::string& ep) {
  PADDLE_ENFORCE_LT(
      *total_time, timeout,
      platform::errors::Unavailable("%s addr=%s timeout, failed reason: %s", op,
                                    ep.c_str(), strerror(errno)));
  ++(*try_times);
  int retry_time = std::min(*try_times * 500, 3000);  // max 3 seconds
  *total_time += retry_time;

  LOG(WARNING) << op << " addr=" << ep << " failed " << *try_times
               << " times with reason: " << strerror(errno) << " retry after "
               << retry_time / 1000.0 << " seconds";
  std::this_thread::sleep_for(std::chrono::milliseconds(retry_time));
}

int CreateListenSocket(const std::string& ep) {
  auto addr = paddle::string::Split(ep, ':');
  PADDLE_ENFORCE_EQ(
      addr.size(), 2UL,
      platform::errors::InvalidArgument(
          "The endpoint should contain host and port, but got %s.", ep));
  std::string host = addr[0];
  int port = std::stoi(addr[1]);

  // creating socket fd
  int server_fd = -1;
  CHECK_SYS_CALL_VAL(socket(AF_INET, SOCK_STREAM, 0), "socket", server_fd);

  // NOTE. Solutions to `Address already in use`.
  // 1. Reuse addr&port. Otherwise, once the server closes the socket
  // before client, the server will enter TIME-WAIT status. If we bind port
  // again, the error `Address already in use` will appear.
  // 2. Or we can close the client first to ensure that the server does
  // not enter the TIME-WAIT state. But this is obviously not as convenient
  // as the reuse method.
  int opt = 1;
#if defined(SO_REUSEPORT)
  // since Linux kernel 3.9
  CHECK_SYS_CALL(setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
                            &opt, sizeof(opt)),
                 "setsockopt");
#else
  CHECK_SYS_CALL(
      setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)),
      "setsockopt");
#endif

  struct sockaddr_in address;
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(port);

  // TODO(wangxi) Set from env, default 900s=15min
  int timeout = 900 * 1000;
  int try_times = 0;
  int total_time = 0;
  while (true) {
    int ret_val = -1;
    RETRY_SYS_CALL_VAL(
        bind(server_fd, (struct sockaddr*)&address, sizeof(address)), "bind",
        ret_val);

    if (ret_val == -1) {
      BindOrConnectFailed(timeout, &try_times, &total_time, "bind", ep);
      continue;
    }
    break;
  }

  CHECK_SYS_CALL(listen(server_fd, 3), "listen");
  LOG(INFO) << "Server listening on: " << ep << " successful.";
  return server_fd;
}

void CloseSocket(int fd) { CHECK_SYS_CALL(close(fd), "close"); }

static int SocketAccept(int server_fd, const char* head) {
  struct sockaddr_in client_addr;
  socklen_t addr_length = sizeof(client_addr);
  char buffer[1024] = {0};
  int conn = -1;

  while (true) {
    CHECK_SYS_CALL_VAL(
        accept(server_fd, reinterpret_cast<struct sockaddr*>(&client_addr),
               &addr_length),
        "accept", conn);

    int ret_val = SocketRecv(conn, buffer, strlen(head));
    if (ret_val > 0 && strncmp(buffer, head, strlen(head)) == 0) {
      break;  // accept client
    } else {
      VLOG(3) << "socket read failed with ret_val=" << ret_val;
      CloseSocket(conn);
    }
  }
  return conn;
}

static int ConnectAddr(const std::string& ep, const char* head) {
  auto addr = paddle::string::Split(ep, ':');
  PADDLE_ENFORCE_EQ(
      addr.size(), 2UL,
      platform::errors::InvalidArgument(
          "The endpoint should contain host and port, but got %s.", ep));
  std::string host = addr[0];
  int port = std::stoi(addr[1]);

  int sock = -1;
  CHECK_SYS_CALL_VAL(socket(AF_INET, SOCK_STREAM, 0), "socket", sock);

  struct sockaddr_in server_addr;
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(port);

  char* ip = NULL;
  struct hostent* hp = NULL;
  hp = gethostbyname(host.c_str());
  PADDLE_ENFORCE_NOT_NULL(hp, platform::errors::InvalidArgument(
                                  "Fail to get host by name %s.", host));

  int i = 0;
  while (hp->h_addr_list[i] != NULL) {
    ip = inet_ntoa(*(struct in_addr*)hp->h_addr_list[i]);
    VLOG(3) << "gethostbyname  host:" << host << "  ->ip: " << ip;
    break;
  }

  PADDLE_ENFORCE_GT(inet_pton(AF_INET, ip, &server_addr.sin_addr), 0,
                    platform::errors::Unavailable("Open address %s failed: %s",
                                                  ep, strerror(errno)));

  // TODO(wangxi) Set from env, default 900s=15min
  int timeout = 900 * 1000;
  int try_times = 0;
  int total_time = 0;
  while (true) {
    int ret_val = -1;
    RETRY_SYS_CALL_VAL(
        connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)),
        "connect", ret_val);

    if (ret_val == -1) {
      BindOrConnectFailed(timeout, &try_times, &total_time, "connect", ep);
      continue;
    }

    CHECK_SYS_CALL(SocketSend(sock, head, strlen(head)), "send");
    break;
  }
  return sock;
}

static void RecvHCCLID(int conn, HcclRootInfo* hccl_id) {
  char buffer[1024] = {0};
  static_assert(HCCL_UNIQUE_ID_BYTES <= 1024,
                "hccl id bytes must <= buffer size");

  CHECK_SYS_CALL(SocketRecv(conn, buffer, HCCL_UNIQUE_ID_BYTES), "recv hccl id");
  memcpy(hccl_id, buffer, HCCL_UNIQUE_ID_BYTES);
}

static void SendHCCLID(int conn, HcclRootInfo* hccl_id) {
  char buffer[1024] = {0};
  memcpy(buffer, hccl_id, HCCL_UNIQUE_ID_BYTES);

  CHECK_SYS_CALL(SocketSend(conn, buffer, HCCL_UNIQUE_ID_BYTES),
                 "send hccl id");
}

void SendBroadCastHCCLID(std::vector<std::string> servers, int hccl_comm_num,
                         std::function<std::string(size_t)> func,
                         const framework::Scope& scope) {
  // connect with server
  std::vector<int> connects;
  for (auto server : servers) {
    VLOG(3) << "connecting endpoint: " << server;
    int conn = ConnectAddr(server, COMM_HEAD);
    connects.push_back(conn);
  }
  VLOG(3) << "connecting completed...";

  for (int i = 0; i < hccl_comm_num; ++i) {
    std::string var_name = func(i);
    auto var = scope.FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::NotFound("Variable with name %s is not found",
                                        var_name.c_str()));
    auto hccl_id = var->GetMutable<HcclRootInfo>();
    PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclGetRootInfo(hccl_id));

    int j = 0;
    for (auto conn : connects) {
      VLOG(3) << "sending hccl_id_var: " << var_name << " to " << servers[j]
              << " hccl_comm_no: " << i;
      SendHCCLID(conn, hccl_id);
      ++j;
    }
    VLOG(3) << "sending completed...";
  }

  // close client
  for (auto conn : connects) {
    CloseSocket(conn);
  }
}

void RecvBroadCastHCCLID(std::string endpoint, int hccl_comm_num,
                         std::function<std::string(size_t)> func,
                         const framework::Scope& scope) {
  int server = CreateListenSocket(endpoint);
  RecvBroadCastHCCLID(server, endpoint, hccl_comm_num, func, scope);
  CloseSocket(server);
}

void RecvBroadCastHCCLID(int server_fd, std::string endpoint, int hccl_comm_num,
                         std::function<std::string(size_t)> func,
                         const framework::Scope& scope) {
  int client = SocketAccept(server_fd, COMM_HEAD);

  for (int i = 0; i < hccl_comm_num; ++i) {
    std::string var_name = func(i);
    auto var = scope.FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::NotFound("Variable with name %s is not found",
                                        var_name.c_str()));
    auto hccl_id = var->GetMutable<HcclRootInfo>();

    VLOG(3) << "trainer: " << endpoint << " receiving hccl_id_var: " << var_name
            << " from trainer 0, hccl_comm_no: " << i;
    RecvHCCLID(client, hccl_id);
  }
  VLOG(3) << "receiving completed...";
  CloseSocket(client);
}

}  // namespace operators
}  // namespace paddle
