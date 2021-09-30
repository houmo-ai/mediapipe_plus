// Copyright 2021 Nanjing Houmo Technology Co.,Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef COMMON_HEADER_FOR_TRT
#define COMMON_HEADER_FOR_TRT
#include <cuda_runtime_api.h>
#include <array>
#include <iostream>
#include <memory>
#include <vector>
#include "assert.h"
#define CUDA_CHECK(callstr)                                               \
  {                                                                       \
    cudaError_t error_code = callstr;                                     \
    if (error_code != cudaSuccess) {                                      \
      std::cerr << "CUDA error " << error_code << ": \""                  \
                << cudaGetErrorString(error_code) << "\" at " << __FILE__ \
                << ":" << __LINE__ << std::endl;                          \
      assert(0);                                                          \
    }                                                                     \
  }

int save_array2file(void* array2save, int size);

namespace mediapipe {
template <typename T>
class TrtBufPacket {
 public:
  TrtBufPacket(size_t elementCount) : mBuffer(nullptr) {
    buffer_size = elementCount * sizeof(T);
    CUDA_CHECK(cudaMalloc((void**)&mBuffer, buffer_size));
  }
  TrtBufPacket(int width, int height, int dep_per_bit) : mBuffer(nullptr) {
    mwidth = width;
    mheight = height;
    mdep_perbit = dep_per_bit;
    buffer_size = width * height * dep_per_bit * sizeof(T);
    CUDA_CHECK(cudaMalloc((void**)&mBuffer, buffer_size));
  }
  TrtBufPacket(T* data_ptr, int data_size) {
    mBuffer = nullptr;
    mHostBuffer = data_ptr;
    buffer_size = data_size;
  }
  TrtBufPacket(const TrtBufPacket& old_packet) {
    mBuffer = old_packet.mBuffer;
    mHostBuffer = old_packet.mHostBuffer;
    buffer_size = old_packet.buffer_size;
    mwidth = old_packet.mwidth;
    mheight = old_packet.mheight;
    mdep_perbit = old_packet.mdep_perbit;
  }

  virtual ~TrtBufPacket() {
#if 0
    if (mBuffer)
    {
        printf("free cuda buffer，mBuffer=0x%x,this=0x%x\r\n",mBuffer,this);
        cudaError_t ret=cudaFree(mBuffer);
        printf("cuda free ret=%d\r\n",ret);
        mBuffer=nullptr;
    }
#endif
  }
  bool copy2device(void* src, int size) const {
    cudaMemcpy(mBuffer, src, size, cudaMemcpyHostToDevice);
  }
  bool copy2host(void* dst, int size) const {
    cudaMemcpy(dst, mBuffer, size, cudaMemcpyDeviceToHost);
  }
  T* GetBuffer() const { return mBuffer; }

 private:
  T* mBuffer;
  T* mHostBuffer;

 public:
  int buffer_size;
  int mwidth;
  int mheight;
  int mdep_perbit;  // color chanels, 1/3/4 or other, for RGBA this value is 4
};

}  // namespace mediapipe

#endif