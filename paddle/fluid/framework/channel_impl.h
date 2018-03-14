/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <stddef.h>  // for size_t
#include <atomic>
#include <condition_variable>
#include <deque>
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

template <typename T>
class ChannelImpl : public paddle::framework::Channel<T> {
  friend Channel<T> *paddle::framework::MakeChannel<T>(size_t);
  friend void paddle::framework::CloseChannel<T>(Channel<T> *);

 public:
  struct QueueMessage {
    T *data;
    std::condition_variable_any cond;
    bool chan_closed = false;
    bool completed = false;
    const void *referrer;   // TODO(thuan): figure out better way to do this
    std::function<void(paddle::framework::Channel<T>*)> callback;

    QueueMessage(T *item) : data(item) {}

    void Wait(std::unique_lock<std::recursive_mutex> &lock) {
      cond.wait(lock, [this]() { return completed; });
    }

    void Notify() {
      completed = true;
      cond.notify_all();
    }
  };

  virtual bool CanSend();
  virtual bool CanReceive();
  virtual bool Send(T *);
  virtual bool Receive(T *);
  virtual size_t Cap() { return cap_; }
  virtual void Lock();
  virtual void Unlock();
  virtual bool IsClosed();
  virtual void Close();
  ChannelImpl(size_t);
  virtual ~ChannelImpl();

  virtual void AddToSendQ(const void *referrer, T* data,
                 std::function<void (paddle::framework::Channel<T>*)> cb);
  virtual void AddToReceiveQ(const void *referrer, T* data,
                 std::function<void (paddle::framework::Channel<T>*)> cb);

  virtual void RemoveFromSendQ(const void *referrer);
  virtual void RemoveFromReceiveQ(const void *referrer);

 private:

  bool send_return(bool value) {
    send_ctr--;
    destructor_cond_.notify_all();
    return value;
  }

  bool recv_return(bool value) {
    recv_ctr--;
    destructor_cond_.notify_all();
    return value;
  }

  size_t cap_;
  std::recursive_mutex mu_;
  bool closed_;
  std::deque<T> buf_;
  std::deque<std::shared_ptr<QueueMessage>> recvq;
  std::deque<std::shared_ptr<QueueMessage>> sendq;
  std::atomic<unsigned> send_ctr{0};
  std::atomic<unsigned> recv_ctr{0};
  std::condition_variable_any destructor_cond_;
};

template <typename T>
ChannelImpl<T>::ChannelImpl(size_t capacity)
    : cap_(capacity), closed_(false), send_ctr(0), recv_ctr(0) {
  PADDLE_ENFORCE_GE(capacity, 0);
}

template <typename T>
bool ChannelImpl<T>::CanSend() {
  std::unique_lock<std::recursive_mutex> lock{mu_};
  return !closed_ && (!recvq.empty() || buf_.size() < cap_);
  lock.unlock();
}

template <typename T>
bool ChannelImpl<T>::CanReceive() {
  std::unique_lock<std::recursive_mutex> lock{mu_};
  return !(closed_ && buf_.empty()) && (!sendq.empty() || buf_.size() > 0);
  lock.unlock();
}

template <typename T>
bool ChannelImpl<T>::Send(T *item) {
  send_ctr++;
  std::unique_lock<std::recursive_mutex> lock{mu_};

  // If channel is closed, do nothing
  if (closed_) {
    lock.unlock();
    // TODO(abhinavarora) Should panic on closed channel
    return send_return(false);
  }

  // If there is a receiver, directly pass the value we want
  // to send to the receiver, bypassing the channel buffer if any
  if (!recvq.empty()) {
    std::shared_ptr<QueueMessage> m = recvq.front();
    recvq.pop_front();
    // Do the data transfer
    *(m->data) = std::move(*item);

    // Execute callback function (if any)
    if (m->callback != nullptr) {
      m->callback(this);
    }

    // Wake up the blocked process and unlock
    m->Notify();
    lock.unlock();
    return send_return(true);
  }

  // Unbuffered channel will always bypass this
  // If buffered channel has space in buffer,
  // write the element to the buffer.
  if (buf_.size() < cap_) {
    // Copy to buffer
    buf_.push_back(std::move(*item));
    // Release lock and return true
    lock.unlock();
    return send_return(true);
  }

  // Block on channel, because some receiver will complete
  // the operation for us
  auto m = std::make_shared<QueueMessage>(item);
  sendq.push_back(m);
  m->Wait(lock);
  // TODO(abhinavarora) Should panic on closed channel
  return send_return(!m->chan_closed);
}

template <typename T>
bool ChannelImpl<T>::Receive(T *item) {
  recv_ctr++;
  std::unique_lock<std::recursive_mutex> lock{mu_};

  // If channel is closed and buffer is empty or
  // channel is unbuffered
  if (closed_ && buf_.empty()) {
    lock.unlock();
    return recv_return(false);
  }

  // If there is a sender, directly receive the value we want
  // from the sender, bypassing the channel buffer if any
  if (!sendq.empty()) {
    std::shared_ptr<QueueMessage> m = sendq.front();
    sendq.pop_front();
    // Do the data transfer
    *item = std::move(*(m->data));

    // Execute callback function (if any)
    if (m->callback != nullptr) {
      m->callback(this);
    }

    // Wake up the blocked process and unlock
    m->Notify();
    lock.unlock();
    return recv_return(true);
  }

  // If this is a buffered channel and there are items in buffer
  if (buf_.size() > 0) {
    // Directly read from buffer
    *item = std::move(buf_.front());
    buf_.pop_front();
    // Release lock and return true
    lock.unlock();
    return recv_return(true);
  }

  // No sender available, block on this channel
  // Some receiver will complete the option for us
  auto m = std::make_shared<QueueMessage>(item);
  recvq.push_back(m);
  m->Wait(lock);

  return recv_return(!m->chan_closed);
}

template <typename T>
void ChannelImpl<T>::Lock() {
  mu_.lock();
}

template <typename T>
void ChannelImpl<T>::Unlock() {
  mu_.unlock();
}

template <typename T>
bool ChannelImpl<T>::IsClosed() {
  std::lock_guard<std::recursive_mutex> lock{mu_};
  return closed_;
}

template <typename T>
void ChannelImpl<T>::Close() {
  std::unique_lock<std::recursive_mutex> lock{mu_};

  if (closed_) {
    // TODO(abhinavarora): closing an already closed channel should panic
    lock.unlock();
    return;
  }

  closed_ = true;

  // Empty the readers
  while (!recvq.empty()) {
    std::shared_ptr<QueueMessage> m = recvq.front();
    recvq.pop_front();
    m->chan_closed = true;

    // Execute callback function (if any)
    if (m->callback != nullptr) {
      m->callback(this);
    }

    m->Notify();
  }

  // Empty the senders
  while (!sendq.empty()) {
    std::shared_ptr<QueueMessage> m = sendq.front();
    sendq.pop_front();
    m->chan_closed = true;

    // Execute callback function (if any)
    if (m->callback != nullptr) {
      m->callback(this);
    }

    m->Notify();
  }
}

template <typename T>
void ChannelImpl<T>::AddToSendQ(const void *referrer, T* data,
                       std::function<void (paddle::framework::Channel<T>*)> cb) {
  std::lock_guard<std::recursive_mutex> lock{mu_};
  auto m = std::make_shared<QueueMessage>(data);
  m->referrer = referrer;
  m->callback = cb;
  sendq.push_back(m);
}

template <typename T>
void ChannelImpl<T>::AddToReceiveQ(const void *referrer, T* data,
                       std::function<void (paddle::framework::Channel<T>*)> cb) {
  std::lock_guard<std::recursive_mutex> lock{mu_};
  auto m = std::make_shared<QueueMessage>(data);
  m->referrer = referrer;
  m->callback = cb;
  recvq.push_back(m);
}

template <typename T>
void ChannelImpl<T>::RemoveFromSendQ(const void *referrer) {
  std::lock_guard<std::recursive_mutex> lock{mu_};

  for (auto it = sendq.begin(); it != sendq.end() ; ) {
    std::shared_ptr<QueueMessage> sendMsg = (std::shared_ptr<QueueMessage>)*it;

    if (sendMsg->referrer == referrer) {
      it = sendq.erase(it);
      send_ctr--;
    } else {
      ++it;
    }
  }
}

template <typename T>
void ChannelImpl<T>::RemoveFromReceiveQ(const void *referrer) {
  std::lock_guard<std::recursive_mutex> lock{mu_};

  for (auto it = recvq.begin(); it != recvq.end() ; ) {
    std::shared_ptr<QueueMessage> recvMsg = (std::shared_ptr<QueueMessage>)*it;

    if (recvMsg->referrer == referrer) {
      it = recvq.erase(it);
      recv_ctr--;
    } else {
      ++it;
    }
  }
}

template <typename T>
ChannelImpl<T>::~ChannelImpl() {
  Close();
  // The destructor must wait for all readers and writers to complete their task
  // The channel has been closed, so we will not accept new readers and writers
  std::unique_lock<std::recursive_mutex> lock{mu_};
  destructor_cond_.wait(lock,
                        [this]() { return send_ctr == 0 && recv_ctr == 0; });
}

}  // namespace framework
}  // namespace paddle
