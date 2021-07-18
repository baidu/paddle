// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/pybind/crypto.h"

#include <memory>
#include <string>

#include "paddle/fluid/framework/io/crypto/aes_cipher.h"
#include "paddle/fluid/framework/io/crypto/cipher.h"
#include "paddle/fluid/framework/io/crypto/cipher_utils.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

using paddle::framework::AESCipher;
using paddle::framework::Cipher;
using paddle::framework::CipherFactory;
using paddle::framework::CipherUtils;

namespace {

class PyCipher : public Cipher {
 public:
  using Cipher::Cipher;
  // encrypt string
  std::string Encrypt(const std::string& plaintext,
                      const std::string& key) override {
    PYBIND11_OVERLOAD_PURE_NAME(std::string, Cipher, "encrypt", Encrypt,
                                plaintext, key);
  }
  // decrypt string
  std::string Decrypt(const std::string& ciphertext,
                      const std::string& key) override {
    PYBIND11_OVERLOAD_PURE_NAME(std::string, Cipher, "decrypt", Decrypt,
                                ciphertext, key);
  }

  // encrypt strings and read them to file,
  void EncryptToFile(const std::string& plaintext, const std::string& key,
                     const std::string& filename) override {
    PYBIND11_OVERLOAD_PURE_NAME(void, Cipher, "encrypt_to_file", EncryptToFile,
                                plaintext, key, filename);
  }
  // read from file and decrypt them
  std::string DecryptFromFile(const std::string& key,
                              const std::string& filename) override {
    PYBIND11_OVERLOAD_PURE_NAME(std::string, Cipher, "decrypt_from_file",
                                DecryptFromFile, key, filename);
  }
};

void BindCipher(py::module* m) {
  py::class_<Cipher, PyCipher, std::shared_ptr<Cipher>>(*m, "Cipher")
      .def(py::init<>())
      .def("encrypt",
           [](Cipher& c, const std::string& plaintext, const std::string& key) {
             std::string ret = c.Encrypt(plaintext, key);
             return py::bytes(ret);
           })
      .def(
          "decrypt",
          [](Cipher& c, const std::string& ciphertext, const std::string& key) {
            std::string ret = c.Decrypt(ciphertext, key);
            return py::bytes(ret);
          })
      .def("encrypt_to_file",
           [](Cipher& c, const std::string& plaintext, const std::string& key,
              const std::string& filename) {
             c.EncryptToFile(plaintext, key, filename);
           })
      .def("decrypt_from_file",
           [](Cipher& c, const std::string& key, const std::string& filename) {
             std::string ret = c.DecryptFromFile(key, filename);
             return py::bytes(ret);
           });
}

void BindAESCipher(py::module* m) {
  py::class_<AESCipher, Cipher, std::shared_ptr<AESCipher>>(*m, "AESCipher")
      .def(py::init<>());
}

void BindCipherFactory(py::module* m) {
  py::class_<CipherFactory>(*m, "CipherFactory")
      .def(py::init<>())
      .def_static("create_cipher",
                  [](const std::string& config_file) {
                    return CipherFactory::CreateCipher(config_file);
                  },
                  py::arg("config_file") = std::string());
}

void BindCipherUtils(py::module* m) {
  py::class_<CipherUtils>(*m, "CipherUtils")
      .def_static("gen_key",
                  [](int length) {
                    std::string ret = CipherUtils::GenKey(length);
                    return py::bytes(ret);
                  })
      .def_static("gen_key_to_file",
                  [](int length, const std::string& filename) {
                    std::string ret =
                        CipherUtils::GenKeyToFile(length, filename);
                    return py::bytes(ret);
                  })
      .def_static("read_key_from_file", [](const std::string& filename) {
        std::string ret = CipherUtils::ReadKeyFromFile(filename);
        return py::bytes(ret);
      });
}

}  // namespace

void BindCrypto(py::module* m) {
  BindCipher(m);
  BindCipherFactory(m);
  BindCipherUtils(m);
  BindAESCipher(m);
}

}  // namespace pybind
}  // namespace paddle
