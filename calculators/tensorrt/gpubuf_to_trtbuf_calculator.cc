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

#include <cuda_runtime_api.h>
#include <array>
#include <memory>
#include <vector>
#include "common_header.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_origin.pb.h"

namespace mediapipe {

#if MEDIAPIPE_DISABLE_GPU
// Just a placeholder to not have to depend on mediapipe::GpuBuffer.
using GpuBuffer = AnyType;
#else
using GpuBuffer = mediapipe::GpuBuffer;
#endif  // MEDIAPIPE_DISABLE_GPU

// Example:
// node {
//   calculator: "GpuBufToTrtBufCalculator"
//   input_stream: "IMAGE_GPU:limited_video_frame"
//   output_stream: "TrtBufPacket"
// }

class GpuBufToTrtBufCalculator : public CalculatorBase {
 private:
  mediapipe::GlCalculatorHelper gpu_helper;
  GLubyte* pixels;
  TrtBufPacket<char>* trtbuf_from_tflite_gpu;

 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<GpuBuffer>();
    cc->Outputs().Index(0).Set<TrtBufPacket<char>>();
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) {
    MP_RETURN_IF_ERROR(gpu_helper.Open(cc));
    pixels = nullptr;
    trtbuf_from_tflite_gpu = nullptr;
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) {
    auto& input = cc->Inputs().Index(0).Get<mediapipe::GpuBuffer>();
    auto texture = gpu_helper.CreateSourceTexture(input);
    LOG(INFO) << "texture.width()=" << texture.width()
              << ";texture.height()=" << texture.height();
    int data_size = texture.width() * texture.height() * 4;
    LOG(INFO) << "data_size=" << data_size;
    if (!pixels) {
      pixels = new GLubyte[data_size];
    }

    GLuint textureObj = texture.name();  // the texture object - glGenTextures
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           textureObj, 0);
    glReadPixels(0, 0, texture.width(), texture.height(), GL_RGBA,
                 GL_UNSIGNED_BYTE, pixels);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(texture.target(), 0);
    glDeleteFramebuffers(1, &fbo);

    if (!trtbuf_from_tflite_gpu) {
      // trtbuf_from_tflite_gpu=new TrtBufPacket<char>(data_size);
      trtbuf_from_tflite_gpu =
          new TrtBufPacket<char>(texture.width(), texture.height(), 4);
    }
    trtbuf_from_tflite_gpu->copy2device((void*)pixels, data_size);
    cc->Outputs().Index(0).AddPacket(
        MakePacket<TrtBufPacket<char>>(*trtbuf_from_tflite_gpu)
            .At(cc->InputTimestamp()));
    return absl::OkStatus();
  }
};

REGISTER_CALCULATOR(GpuBufToTrtBufCalculator);
}  // namespace mediapipe
