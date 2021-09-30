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
#include "common/argsParser.h"
#include "common/buffers.h"
#include "common/common.h"
#include "common/logger.h"
#include "common/parserOnnxConfig.h"

#include <math.h>
#include <stdio.h>
#include <cstdarg>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "common_header.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

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
using namespace samplesCommon;

#if MEDIAPIPE_DISABLE_GPU
// Just a placeholder to not have to depend on mediapipe::GpuBuffer.
using GpuBuffer = AnyType;
#else
using GpuBuffer = mediapipe::GpuBuffer;
#endif  // MEDIAPIPE_DISABLE_GPU

// Example:
// node {
//   calculator: "TrtInferenceCalculator"
//   input_stream: "IMAGE_GPU:limited_video_frame"
//   output_stream: "vector:output_feature"
// }

// TRT:Build->Infer
// Cal: Build in Open
//      Infer in Process
class TrtInferenceCalculator : public CalculatorBase {
 private:
  mediapipe::GlCalculatorHelper gpu_helper;
  GLubyte* pixels;
  samplesCommon::OnnxSampleParams mParams;  //!< The parameters for the sample.

  nvinfer1::Dims mInputDims;   //!< The dimensions of the input to the network.
  nvinfer1::Dims mOutputDims;  //!< The dimensions of the output to the network.
  nvinfer1::Dims
      mOutputDims1;  //!< The dimensions of the output to the network.
  int mNumber{0};    //!< The number to classify
  std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
  SampleUniquePtr<nvinfer1::IExecutionContext> context;
  std::vector<void*> mDeviceBindings;
  std::vector<int> output_nbsize;
  void* mBuffer;
  void* mBuffer_out;
  bool bind_out_buffer;
  cv::VideoWriter writer;
  vector<unsigned char*> dst_pts;  // output points
  std::vector<unsigned char*> out_features;
  cv::Mat raw_image;

 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<TrtBufPacket<char>>();
    cc->Outputs().Index(0).Set<std::vector<unsigned char*>>();
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) {
    MP_RETURN_IF_ERROR(gpu_helper.Open(cc));
    // construct trt enging
    mParams.dataDirs.push_back("./models");
    mParams.onnxFileName = "face_detection_front-micro-10.onnx";
    mParams.inputTensorNames.push_back("input");
    mParams.outputTensorNames.push_back("regressors");
    mParams.outputTensorNames.push_back("classificators");
    mParams.dlaCore = 0;  // args.useDLACore;
    mParams.int8 = 0;     // args.runInInt8;
    mParams.fp16 = 0;     // args.runInFp16;
    mBuffer = nullptr;
    bind_out_buffer = false;
    dst_pts.clear();
    dst_pts.shrink_to_fit();
    build();
    CUDA_CHECK(cudaMalloc((void**)&mBuffer,
                          mInputDims.d[1] * mInputDims.d[2] * 3 *
                              4));  // 3 chanels , 4bytes per float value
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) {
    static int frame_count = 0;
    auto& input = cc->Inputs().Index(0).Get<TrtBufPacket<char>>();

    // only for test back edge
    // cc->Outputs().Index(1).AddPacket(cc->Inputs().Index(0).Value());
    cv::Mat input_image(input.mheight, input.mwidth,
                        (4 == input.mdep_perbit) ? CV_8UC4 : CV_8UC3);
    cudaMemcpy(input_image.data, input.GetBuffer(),
               input.mwidth * input.mheight * input.mdep_perbit,
               cudaMemcpyDeviceToHost);
    cv::cvtColor(input_image, raw_image, cv::COLOR_RGBA2RGB);

    if (frame_count % 100 == 0) {
      cv::imwrite("trtcalculator_get.jpg", raw_image);
    }
    frame_count++;
    prepare_in_out_buffer(mEngine, input.GetBuffer(), raw_image);
    bool status = context->executeV2(mDeviceBindings.data());
    if (!status) {
      LOG(ERROR) << "executeV2 failed";
    } else {
      out_features.clear();
      out_features.shrink_to_fit();
      for (int i = 0; i < mEngine->getNbBindings(); i++) {
        if (mEngine->bindingIsInput(i)) {
          LOG(INFO) << "omit input:" << i;
          continue;
        }
        if (0 == dst_pts.size()) {
          dst_pts.resize(mEngine->getNbBindings());
          for (int index1 = 0; index1 < mEngine->getNbBindings(); index1++) {
            dst_pts[index1] = nullptr;
          }
        }
        if (!dst_pts[i]) {
          dst_pts[i] = (unsigned char*)malloc(output_nbsize[i]);
          LOG(INFO) << "--malloc dst" << i;
        }
        cudaError_t ret = cudaMemcpy(dst_pts[i], mDeviceBindings[i],
                                     output_nbsize[i], cudaMemcpyDeviceToHost);
        LOG(INFO) << "cudaMemcpy ret" << ret;
        out_features.push_back(dst_pts[i]);
// printf("--output_nbsize[%d]=%d,
// mDeviceBindings[%d]=0x%x\r\n",i,output_nbsize[i],i, mDeviceBindings[i]);
#ifdef PRINT_OUT_FEATURE
        for (int index1 = 0; index1 < 50; index1++) {
          float* out_data = (float*)dstPtr;
          printf("%f\r\n", out_data[index1]);
        }
#endif
      }
      out_features.push_back((unsigned char*)&raw_image);
      cc->Outputs().Index(0).AddPacket(
          MakePacket<std::vector<unsigned char*>>(out_features)
              .At(cc->InputTimestamp()));
    }
    return absl::OkStatus();
  }

  bool prepare_in_out_buffer(
      std::shared_ptr<nvinfer1::ICudaEngine> engine, void* mBuffer_in,
      cv::Mat& raw_image, const int batchSize = 0,
      const nvinfer1::IExecutionContext* context = nullptr) {
    int mBatchSize = batchSize;
    // Full Dims implies no batch size.
    assert(engine->hasImplicitBatchDimension() || mBatchSize == 0);
    // Create host and device buffers
    for (int i = 0; i < mEngine->getNbBindings(); i++) {
      auto dims = context ? context->getBindingDimensions(i)
                          : mEngine->getBindingDimensions(i);
      size_t vol = context || !mBatchSize ? 1 : static_cast<size_t>(mBatchSize);
      nvinfer1::DataType type = mEngine->getBindingDataType(i);
      int vecDim = mEngine->getBindingVectorizedDim(i);
      if (-1 != vecDim)  // i.e., 0 != lgScalarsPerVector
      {
        int scalarsPerVec = mEngine->getBindingComponentsPerElement(i);
        dims.d[vecDim] = divUp(dims.d[vecDim], scalarsPerVec);
        vol *= scalarsPerVec;
      }
      vol *= samplesCommon::volume(dims);
      if (mEngine->bindingIsInput(i)) {
        int INPUT_SIZE = mInputDims.d[1];
        if (!bind_out_buffer) {
          mDeviceBindings.emplace_back(mBuffer);
          output_nbsize.emplace_back(INPUT_SIZE * INPUT_SIZE * 3 * 4);
        }
// Feed Data
#if SINGLE_PIC
        cv::Mat raw_image_BGR = cv::imread(
            "/media/houmo/490fd486-af35-4236-aea5-f16fde9df9b5/zhenjiang/"
            "mediapipe/image.jpg");
        cv::Mat raw_image;
        cv::cvtColor(raw_image_BGR, raw_image, cv::COLOR_BGR2RGB);
#endif

        int raw_image_height = raw_image.rows;
        int raw_image_width = raw_image.cols;
        cv::Mat image;
        cv::resize(raw_image, image, cv::Size(INPUT_SIZE, INPUT_SIZE));
        image.convertTo(image, CV_32FC3);
        image = (image * 2 / 255.0f) - 1;
        if (image.isContinuous()) {
          LOG(INFO) << "++cv mat is continuous";
        } else {
          LOG(INFO) << "--cv mat is not continuous";
        }
        cudaError_t ret =
            cudaMemcpy(mBuffer, image.data, INPUT_SIZE * INPUT_SIZE * 3 * 4,
                       cudaMemcpyHostToDevice);
        LOG(INFO) << "++cudaMemcpy ret=" << ret;
      } else if (!bind_out_buffer) {
        int buffe_size = vol * sizeof(float);
        mBuffer_out = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&mBuffer_out, buffe_size));
        if (mBuffer_out) {
          mDeviceBindings.emplace_back(mBuffer_out);
          output_nbsize.emplace_back(buffe_size);
          LOG(INFO) << "cudamalloc success mem size:" << buffe_size;
        } else {
          LOG(ERROR) << "cudamalloc mBuffer_out failed";
        }
      }
    }
    bind_out_buffer = true;
  }

  bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                        SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
                        SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                        SampleUniquePtr<nvonnxparser::IParser>& parser) {
    auto parsed = parser->parseFromFile(
        locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed) {
      return false;
    }
    config->setMaxWorkspaceSize(16_MiB);
    if (mParams.fp16) {
      config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8) {
      config->setFlag(BuilderFlag::kINT8);
      samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    }
    // Disable USE DLA
    // samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
    return true;
  }

  bool build() {
    LOG(INFO) << "++++in build before execute";
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;
    auto t1 = high_resolution_clock::now();
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder) {
      return false;
    }

    const auto explicitBatch =
        1U << static_cast<uint32_t>(
            NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicitBatch));
    if (!network) {
      return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    if (!config) {
      return false;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser) {
      return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed) {
      return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream) {
      return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{
        builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
      return false;
    }

    SampleUniquePtr<IRuntime> runtime{
        createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime) {
      return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()),
        samplesCommon::InferDeleter());
    if (!mEngine) {
      return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    // ASSERT(mInputDims.nbDims == 4);
    // printf("mInputDims.nbDims=%d\r\n",mInputDims.nbDims);

    ASSERT(network->getNbOutputs() == 2);
    mOutputDims = network->getOutput(0)->getDimensions();
    mOutputDims1 = network->getOutput(1)->getDimensions();
    // printf("mOutputDims.nbDims=%d\r\n",mOutputDims.nbDims);
    // ASSERT(mOutputDims.nbDims == 2);

    context = SampleUniquePtr<nvinfer1::IExecutionContext>(
        mEngine->createExecutionContext());
    LOG(INFO) << "---in build after execute";
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    LOG(INFO) << "###Build Time::" << ms_double.count() << "ms";
    return true;
  }
};

REGISTER_CALCULATOR(TrtInferenceCalculator);
}  // namespace mediapipe
