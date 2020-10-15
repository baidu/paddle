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

#pragma once

#include <time.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h
#include "glog/logging.h"

#if !defined(_WIN32)
#include <dlfcn.h>  // dladdr
#include <sys/stat.h>
#include <sys/time.h>
#include <algorithm>  // std::accumulate
#else
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
// solve static linking error in windows
// https://github.com/google/glog/issues/301
#define GOOGLE_GLOG_DLL_DECL
#include <io.h>  // _popen, _pclose
#include <stdio.h>
#include <windows.h>
#include <numeric>  // std::accumulate in msvc
#ifndef S_ISDIR     // windows port for sys/stat.h
#define S_ISDIR(mode) (((mode)&S_IFMT) == S_IFDIR)
#endif  // S_ISDIR

static void *dlsym(void *handle, const char *symbol_name) {
  FARPROC found_symbol;
  found_symbol = GetProcAddress((HMODULE)handle, symbol_name);

  if (found_symbol == NULL) {
    throw std::runtime_error(std::string(symbol_name) + " not found.");
  }
  return reinterpret_cast<void *>(found_symbol);
}

static void *dlopen(const char *filename, int flag) {
  std::string file_name(filename);
  HMODULE hModule = LoadLibrary(file_name.c_str());
  if (!hModule) {
    throw std::runtime_error(file_name + " not found.");
  }
  return reinterpret_cast<void *>(hModule);
}

static int gettimeofday(struct timeval *tp, void *tzp) {
  time_t clock;
  struct tm tm;
  SYSTEMTIME wtm;

  GetLocalTime(&wtm);
  tm.tm_year = wtm.wYear - 1900;
  tm.tm_mon = wtm.wMonth - 1;
  tm.tm_mday = wtm.wDay;
  tm.tm_hour = wtm.wHour;
  tm.tm_min = wtm.wMinute;
  tm.tm_sec = wtm.wSecond;
  tm.tm_isdst = -1;
  clock = mktime(&tm);
  tp->tv_sec = clock;
  tp->tv_usec = wtm.wMilliseconds * 1000;

  return (0);
}
#endif  // !_WIN32

static void ExecShellCommand(const std::string &cmd, std::string *message) {
  char buffer[128];
#if !defined(_WIN32)
  std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
#else
  std::shared_ptr<FILE> pipe(_popen(cmd.c_str(), "r"), _pclose);
#endif  // _WIN32
  if (!pipe) {
    LOG(ERROR) << "error running command: " << cmd;
    return;
  }
  while (!feof(pipe.get())) {
    if (fgets(buffer, 128, pipe.get()) != nullptr) {
      *message += buffer;
    }
  }
}

static bool PathExists(const std::string &path) {
#if !defined(_WIN32)
  struct stat statbuf;
  if (stat(path.c_str(), &statbuf) != -1) {
    if (S_ISDIR(statbuf.st_mode)) {
      return true;
    }
  }
#else
  struct _stat statbuf;
  if (_stat(path.c_str(), &statbuf) != -1) {
    if (S_ISDIR(statbuf.st_mode)) {
      return true;
    }
  }
#endif  // !_WIN32
  return false;
}

// TODO(yuyang18): If the functions below are needed by other files, move them
// to paddle::filesystem namespace.
#if !defined(_WIN32)
constexpr char kSEP = '/';
#else
constexpr char kSEP = '\\';
#endif  // _WIN32

static bool FileExists(const std::string &filepath) {
#if !defined(_WIN32)
  struct stat buffer;
  return (stat(filepath.c_str(), &buffer) == 0);
#else
  struct _stat buffer;
  return (_stat(filepath.c_str(), &buffer) == 0);
#endif  // !_WIN32
}

static std::string DirName(const std::string &filepath) {
  auto pos = filepath.rfind(kSEP);
  if (pos == std::string::npos) {
    return "";
  }
  return filepath.substr(0, pos);
}

static void MkDir(const char *path) {
  std::string path_error(path);
  path_error += " mkdir failed!";
#if !defined(_WIN32)
  if (mkdir(path, 0755)) {
    if (errno != EEXIST) {
      throw std::runtime_error(path_error);
    }
  }
#else
  BOOL return_value = CreateDirectory(path, NULL);
  if (!return_value) {
    auto errorno = GetLastError();
    if (errorno != ERROR_ALREADY_EXISTS) {
      throw std::runtime_error(path_error);
    }
  }
#endif  // !_WIN32
}

static void MkDirRecursively(const char *fullpath) {
  if (*fullpath == '\0') return;  // empty string
  if (FileExists(fullpath)) return;

  MkDirRecursively(DirName(fullpath).c_str());
  MkDir(fullpath);
}
