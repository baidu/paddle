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

#include <cryptopp/aes.h>
#include <cryptopp/cryptlib.h>
#include <cryptopp/files.h>
#include <cryptopp/filters.h>
#include <cryptopp/gcm.h>
#include <cryptopp/osrng.h>
#include <cryptopp/secblock.h>

#include <algorithm>
#include <fstream>
#include <streambuf>
#include <string>

namespace paddle {
namespace framework {

class CryptFilebuf : public std::filebuf {
 public:
  // normal ctr filebuf, same with std::filebuf
  CryptFilebuf()
      : std::filebuf(),
        _is_end(false),
        _fs(nullptr),
        _ef(nullptr),
        _df{nullptr} {}
  // encryption filebuf ctr
  explicit CryptFilebuf(CryptoPP::AuthenticatedEncryptionFilter* ef)
      : std::filebuf(), _is_end(false), _fs(nullptr), _ef(ef), _df{nullptr} {}
  // decryption filebuf ctr
  explicit CryptFilebuf(CryptoPP::AuthenticatedDecryptionFilter* df)
      : std::filebuf(), _is_end(false), _ef(nullptr), _df(df) {}

  ~CryptFilebuf() { this->close(); }

 protected:
  std::streamsize xsgetn(char_type* s, std::streamsize n) override {
    if (!(_ef || _df)) {
      // normal xsgetn, same with std::filebuf::xsgetn()
      return std::filebuf::xsgetn(s, n);
    } else {
      // decryption filebuf
      std::string plain;
      try {
        std::string cipher;
        cipher.resize(n);
        std::filebuf::xsgetn(&(cipher.at(0)), n);
        _df->ChannelPut(CryptoPP::DEFAULT_CHANNEL,
                        (const CryptoPP::byte*)cipher.data(), n);
        _df->SetRetrievalChannel(CryptoPP::DEFAULT_CHANNEL);
        plain.resize(n);
        _df->Get((CryptoPP::byte*)plain.data(), n);
        traits_type::copy(s, plain.c_str(), n);
        // whether is last read
        const size_t TAG_SIZE = 16;
        auto cur_pos = this->pubseekoff(0, std::ios::cur, std::ios::in);
        this->pubseekoff(TAG_SIZE, std::ios::cur, std::ios::in);
        if (traits_type::eq_int_type(sgetc(), traits_type::eof())) {
          _is_end = true;
        } else {
          this->pubseekpos(cur_pos, std::ios::in);
        }
        if (_is_end) {
          // check file's integrity
          _df->ChannelMessageEnd(CryptoPP::DEFAULT_CHANNEL);
          bool b = _df->GetLastResult();
          if (!b) {
            // throw std::runtime_error("integrity check failed!");
            throw CryptoPP::HashVerificationFilter::HashVerificationFailed();
          }
        }
      } catch (CryptoPP::InvalidArgument& e) {
        std::cerr << "decryption error: CryptoPP-InvalidArgument." << std::endl;
        std::cerr << e.what() << std::endl;
        throw;
      } catch (CryptoPP::AuthenticatedSymmetricCipher::BadState& e) {
        std::cerr << "decryption error: CryptoPP-BadState." << std::endl;
        std::cerr << e.what() << std::endl;
        throw;
      } catch (CryptoPP::HashVerificationFilter::HashVerificationFailed& e) {
        std::cerr << "decryption error: "
                  << "CryptoPP-HashVerificationFailed." << std::endl;
        std::cerr << e.what() << std::endl;
        throw;
      }
      return n;
    }
  }
  std::streamsize xsputn(const char_type* s, std::streamsize n) override {
    if (!(_ef || _df)) {
      // not a secure buf
      return std::filebuf::xsputn(s, n);
    } else {
      // enc buf
      try {
        if (_is_end) {
          _ef->ChannelMessageEnd(CryptoPP::DEFAULT_CHANNEL);
        } else {
          std::string plain(s, s + n);
          _ef->ChannelPut(CryptoPP::DEFAULT_CHANNEL,
                          (const CryptoPP::byte*)plain.data(), n);
        }
      } catch (CryptoPP::BufferedTransformation::NoChannelSupport& e) {
        std::cerr << "encryption error: "
                  << "CryptoPP-NoChannelSupport." << std::endl;
        std::cerr << e.what() << std::endl;
        throw;
      } catch (CryptoPP::AuthenticatedSymmetricCipher::BadState& e) {
        std::cerr << "encryption error: CryptoPP-BadState." << std::endl;
        std::cerr << e.what() << std::endl;
        throw;
      } catch (CryptoPP::InvalidArgument& e) {
        std::cerr << "encryption error: "
                  << "CryptoPP-InvalidArgument." << std::endl;
        std::cerr << e.what() << std::endl;
        throw;
      }
      return n;
    }
  }

  int_type sync() override { return std::filebuf::sync(); }

  int_type overflow(int_type c = traits_type::eof()) {
    return std::filebuf::overflow(c);
  }

 public:
  void post_process() {
    if (_ef) {
      _is_end = true;
      xsputn("", 0);
    }
    // if (_df && !_is_end) {
    // had process in xsgetn()
    //}
  }

 private:
  bool _is_end;
  CryptoPP::FileSink* _fs;
  CryptoPP::AuthenticatedEncryptionFilter* _ef;
  CryptoPP::AuthenticatedDecryptionFilter* _df;
};

class OfstreamExt : public std::ostream {
 public:
  OfstreamExt()
      : std::ostream(),
        _fs(nullptr),
        _e(nullptr),
        _ef(nullptr),
        _fout(nullptr) {
    _fbuf = new CryptFilebuf();
    this->init(_fbuf);
  }

  explicit OfstreamExt(const char* s,
                       std::ios_base::openmode mode = std::ios_base::out)
      : std::ostream(),
        _fs(nullptr),
        _e(nullptr),
        _ef(nullptr),
        _fout(nullptr) {
    _fbuf = new CryptFilebuf();
    this->init(_fbuf);
    this->open(s, mode);
  }

  explicit OfstreamExt(const char* s, std::ios_base::openmode mode,
                       bool sec_enhanced, unsigned char* key_str,
                       size_t key_len, const int TAG_SIZE)
      : std::ostream() {
    if (sec_enhanced) {
      CryptoPP::AutoSeededRandomPool prng;
      CryptoPP::SecByteBlock key(key_str, key_len);
      CryptoPP::byte iv[12];
      prng.GenerateBlock(iv, sizeof(iv));
      _fout = new std::ofstream(s, mode);
      // write iv first
      _fout->write((const char*)iv, sizeof(iv));
      _e = new CryptoPP::GCM<CryptoPP::AES>::Encryption();
      _e->SetKeyWithIV(key, key_len, iv, sizeof(iv));
      _fs = new CryptoPP::FileSink(*_fout);
      _ef = new CryptoPP::AuthenticatedEncryptionFilter(*_e, _fs, false,
                                                        TAG_SIZE);
      _fbuf = new CryptFilebuf(_ef);
    } else {
      _e = nullptr;
      _fs = nullptr;
      _ef = nullptr;
      _fbuf = new CryptFilebuf();
      _fout = nullptr;
    }
    this->init(_fbuf);
    this->open(s, mode);
  }

  ~OfstreamExt() {
    if (!_is_close) {
      this->close();
    }
    if (_e != nullptr) {
      delete _fout;
      delete _e;
      delete _ef;
    }
    delete _fbuf;
  }

  std::filebuf* rdbuf() const { return const_cast<CryptFilebuf*>(_fbuf); }

  bool is_open() { return _fbuf->is_open(); }

  bool is_open() const { return _fbuf->is_open(); }

  void open(const char* s, std::ios_base::openmode mode = std::ios_base::out) {
    if (!_fbuf->open(s, mode | std::ios_base::out)) {
      this->setstate(ios_base::failbit);
    } else {
      this->clear();
    }
  }

  void open(const std::string& s,
            std::ios_base::openmode mode = std::ios_base::out) {
    if (!_fbuf->open(s, mode | std::ios_base::out)) {
      this->setstate(std::ios_base::failbit);
    } else {
      this->clear();
    }
  }

  void close() {
    _is_close = true;
    _fbuf->post_process();
    if (!_fbuf->close()) {
      this->setstate(std::ios_base::failbit);
    }
  }

 private:
  CryptFilebuf* _fbuf;
  CryptoPP::FileSink* _fs;
  CryptoPP::GCM<CryptoPP::AES>::Encryption* _e;
  CryptoPP::AuthenticatedEncryptionFilter* _ef;
  std::ofstream* _fout;
  bool _is_close{false};
};

// notice: integrity checking work only when
// all ciphertexts in file had been read comppletely
class IfstreamExt : public std::istream {
 public:
  IfstreamExt() : std::istream(), _d(nullptr), _df(nullptr) {
    _fbuf = new CryptFilebuf();
    this->init(_fbuf);
  }

  explicit IfstreamExt(const char* s,
                       std::ios_base::openmode mode = std::ios_base::in)
      : std::istream(), _d(nullptr), _df(nullptr) {
    _fbuf = new CryptFilebuf();
    this->init(_fbuf);
    this->open(s, mode);
  }

  explicit IfstreamExt(const char* s, std::ios_base::openmode mode,
                       bool sec_enhanced, unsigned char* key_str,
                       size_t key_len, const int TAG_SIZE)
      : std::istream() {
    if (sec_enhanced) {
      // data structure in s: IV || ciphertexts || MAC
      std::ifstream fin(s, mode);
      CryptoPP::SecByteBlock key(key_str, key_len);
      CryptoPP::byte iv[12];
      // read iv
      fin.read(reinterpret_cast<char*>(iv), sizeof(iv));
      auto cipher_pos = fin.tellg();

      _d = new CryptoPP::GCM<CryptoPP::AES>::Decryption();
      _d->SetKeyWithIV(key, key_len, iv, sizeof(iv));
      _df = new CryptoPP::AuthenticatedDecryptionFilter(
          *_d, NULL,
          CryptoPP::AuthenticatedDecryptionFilter::MAC_AT_BEGIN |
              CryptoPP::AuthenticatedDecryptionFilter::THROW_EXCEPTION,
          TAG_SIZE);

      _fbuf = new CryptFilebuf(_df);
      this->init(_fbuf);
      this->open(s, mode);

      this->seekg(0, std::ios::end);
      auto mac_pos = tellg() - (int64_t)TAG_SIZE;
      char mac[TAG_SIZE];

      fin.seekg(mac_pos, std::ios::beg);
      fin.read(mac, TAG_SIZE);
      _df->ChannelPut(CryptoPP::DEFAULT_CHANNEL, (const CryptoPP::byte*)mac,
                      TAG_SIZE);
      seekg(cipher_pos, std::ios::beg);

    } else {
      _d = nullptr;
      _fbuf = new CryptFilebuf();
      this->init(_fbuf);
      this->open(s, mode);
    }
  }

  ~IfstreamExt() {
    if (_d != nullptr) {
      delete _d;
      delete _df;
    }
    delete _fbuf;
  }

  std::filebuf* rdbuf() const { return const_cast<CryptFilebuf*>(_fbuf); }

  bool is_open() { return _fbuf->is_open(); }

  bool is_open() const { return _fbuf->is_open(); }

  void open(const char* s, std::ios_base::openmode mode = std::ios_base::in) {
    if (!_fbuf->open(s, mode | std::ios_base::in)) {
      this->setstate(std::ios_base::failbit);
    } else {
      this->clear();
    }
  }

  void open(const std::string& s,
            std::ios_base::openmode mode = std::ios_base::in) {
    if (!_fbuf->open(s, mode | std::ios_base::in)) {
      this->setstate(std::ios_base::failbit);
    } else {
      this->clear();
    }
  }

  void close() {
    if (!_fbuf->close()) {
      this->setstate(std::ios_base::failbit);
    }
  }

 private:
  CryptFilebuf* _fbuf;
  CryptoPP::GCM<CryptoPP::AES>::Decryption* _d;
  CryptoPP::AuthenticatedDecryptionFilter* _df;
};

}  // namespace framework
}  // namespace paddle
