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
#include <typeindex>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

// Channel is the abstract class of buffered and un-buffered channels.
template <typename T>
class Channel {
 public:
  virtual bool CanSend() = 0;
  virtual bool CanReceive() = 0;
  virtual bool Send(T*) = 0;
  virtual bool Receive(T*) = 0;
  virtual size_t Cap() = 0;
  virtual void Lock() = 0;
  virtual void Unlock() = 0;
  virtual bool IsClosed() = 0;
  virtual void Close() = 0;
  virtual ~Channel() {}

  virtual void AddToSendQ(const void *referrer, T* data,
                 std::function<void (paddle::framework::Channel<T>*)> cb) = 0;
  virtual void AddToReceiveQ(const void *referrer, T* data,
                 std::function<void (paddle::framework::Channel<T>*)> cb) = 0;
  virtual void RemoveFromSendQ(const void *referrer) = 0;
  virtual void RemoveFromReceiveQ(const void *referrer) = 0;
};

// Forward declaration of channel implementations.
template <typename T>
class ChannelImpl;

template <typename T>
Channel<T>* MakeChannel(size_t buffer_size) {
  return new ChannelImpl<T>(buffer_size);
}

template <typename T>
void CloseChannel(Channel<T>* ch) {
  ch->Close();
}

/*
 * The ChannelHolder class serves two main purposes:
 * 1. It acts as a unified wrapper for the different kinds of
 *    channels, i.e. Buffered and Unbuffered channels. This is
 *    similar to the ReaderHolder class.
 * 2. It also helps us in TypeHiding. This is similar to the
 *    PlaceHolder implementations in variable.h and tensor.h.
 */
class ChannelHolder {
 public:
  template <typename T>
  void Reset(size_t buffer_size) {
    holder_.reset(new PlaceholderImpl<T>(buffer_size));
  }

//  template <typename T>
//  bool CanSend() {
//    if (!IsInitialized()) return false;
//    // Static cast should be safe because we have ensured that types are same
//    Channel<T>* channel = static_cast<Channel<T>*>(holder_->Ptr());
//    return channel != nullptr ? channel->CanSend() : false;
//  }
//
//  template <typename T>
//  bool CanReceive() {
//    if (!IsInitialized()) return false;
//    // Static cast should be safe because we have ensured that types are same
//    Channel<T>* channel = static_cast<Channel<T>*>(holder_->Ptr());
//    return channel != nullptr ? channel->CanReceive() : false;
//  }

  template <typename T>
  bool Send(T* data) {
    if (!IsInitialized()) return false;
    PADDLE_ENFORCE_EQ(holder_->Type(), std::type_index(typeid(T)));
    // Static cast should be safe because we have ensured that types are same
    Channel<T>* channel = static_cast<Channel<T>*>(holder_->Ptr());
    return channel != nullptr ? channel->Send(data) : false;
  }

  template <typename T>
  bool Receive(T* data) {
    if (!IsInitialized()) return false;
    PADDLE_ENFORCE_EQ(holder_->Type(), std::type_index(typeid(T)));
    Channel<T>* channel = static_cast<Channel<T>*>(holder_->Ptr());
    return channel != nullptr ? channel->Receive(data) : false;
  }

  bool IsClosed() {
    if (IsInitialized()) {
      return holder_->IsClosed();
    }
    return false;
  }

  bool CanSend() {
    if (IsInitialized()) {
      return holder_->CanSend();
    }
    return false;
  }

  bool CanReceive() {
    if (IsInitialized()) {
      return holder_->CanReceive();
    }
    return false;
  }

  void close() {
    if (IsInitialized()) holder_->Close();
  }

  size_t Cap() {
    if (IsInitialized()) return holder_->Cap();
    return -1;
  }

  void Lock() {
    if (IsInitialized()) holder_->Lock();
  }

  void Unlock() {
    if (IsInitialized()) holder_->Unlock();
  }

  template <typename T>
  void AddToSendQ(const void *referrer, T* data,
         std::function<void (paddle::framework::Channel<T>*)> cb) {
    if (IsInitialized()) {
      Channel<T>* channel = static_cast<Channel<T>*>(holder_->Ptr());
      if (channel != nullptr) {
        channel->AddToSendQ(referrer, data, cb);
      }
    }
  }

  template <typename T>
  void AddToReceiveQ(const void *referrer, T* data,
         std::function<void (paddle::framework::Channel<T>*)> cb) {
    if (IsInitialized()) {
      Channel<T>* channel = static_cast<Channel<T>*>(holder_->Ptr());
      if (channel != nullptr) {
        channel->AddToReceiveQ(referrer, data, cb);
      }
    }
  }

  template <typename T>
  void RemoveFromSendQ(const void *referrer) {
    if (IsInitialized()) {
      Channel<T>* channel = static_cast<Channel<T>*>(holder_->Ptr());
      if (channel != nullptr) {
        channel->RemoveFromSendQ(referrer);
      }
    }
  }

  template <typename T>
  void RemoveFromReceiveQ(const void *referrer) {
    if (IsInitialized()) {
      Channel<T>* channel = static_cast<Channel<T>*>(holder_->Ptr());
      if (channel != nullptr) {
        channel->RemoveFromReceiveQ(referrer);
      }
    }
  }

  inline bool IsInitialized() const { return holder_ != nullptr; }

  inline const std::type_index Type() {
    PADDLE_ENFORCE_EQ(IsInitialized(), true);
    return holder_->Type();
  }

 private:
  /**
   * @note    Placeholder hides type T, so it doesn't appear as a template
   *          parameter of ChannelHolder.
   */
  struct Placeholder {
    virtual ~Placeholder() {}
    virtual const std::type_index Type() const = 0;
    virtual void* Ptr() const = 0;
    virtual bool IsClosed() = 0;
    virtual bool CanSend() = 0;
    virtual bool CanReceive() = 0;
    virtual void Close() = 0;
    virtual void Lock() = 0;
    virtual void Unlock() = 0;
    virtual size_t Cap() = 0;
  };

  template <typename T>
  struct PlaceholderImpl : public Placeholder {
    PlaceholderImpl(size_t buffer_size) : type_(std::type_index(typeid(T))) {
      channel_.reset(MakeChannel<T>(buffer_size));
    }

    virtual const std::type_index Type() const { return type_; }

    virtual void* Ptr() const { return static_cast<void*>(channel_.get()); }

    virtual bool IsClosed() {
      if (channel_) {
        return channel_->IsClosed();
      }
      return false;
    }

    virtual bool CanSend() {
      if (channel_) {
        return channel_->CanSend();
      }
      return false;
    }

    virtual bool CanReceive() {
      if (channel_) {
        return channel_->CanReceive();
      }
      return false;
    }

    virtual void Close() {
      if (channel_) channel_->Close();
    }

    virtual size_t Cap() {
      if (channel_)
        return channel_->Cap();
      else
        return -1;
    }

    virtual void Lock() {
      if (channel_) channel_->Lock();
    }

    virtual void Unlock() {
      if (channel_) channel_->Unlock();
    }

    std::unique_ptr<Channel<T>> channel_;
    const std::type_index type_;
  };

  // Pointer to a PlaceholderImpl object
  std::unique_ptr<Placeholder> holder_;
};

}  // namespace framework
}  // namespace paddle

#include "paddle/fluid/framework/channel_impl.h"
