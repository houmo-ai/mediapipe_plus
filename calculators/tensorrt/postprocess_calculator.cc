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

#include <array>
#include <memory>
#include <vector>
#include <cuda_runtime_api.h>
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

float anchors[1000][4];

bool init_anchor() {
  std::string line;
  std::ifstream in("./anchor.txt");
  if (in) {
    int ln = 0;
    while (getline(in, line)) {
      std::vector<std::string> res;
      std::string result;
      std::stringstream input;
      input << line;
      while (input >> result) res.push_back(result);
      for (int j = 0; j < res.size(); j++) {
        anchors[ln][j] = std::stof(res[j]);
      }
      ln++;
    }
  }
  return true;
}

struct Object {
  cv::Rect_<float> rect;
  int label;
  std::vector<cv::Point2f> landmarks;
  float prob;
};

inline int sprintf_s(char* buffer, size_t sizeOfBuffer, const char* format,
                     ...) {
  va_list ap;
  va_start(ap, format);
  int result = vsnprintf(buffer, sizeOfBuffer, format, ap);
  va_end(ap);
  return result;
}

template <size_t sizeOfBuffer>
inline int sprintf_s(char (&buffer)[sizeOfBuffer], const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  int result = vsnprintf(buffer, sizeOfBuffer, format, ap);
  va_end(ap);
  return result;
}

float sigmoid(float x) { return (1 / (1 + exp(-x))); }
std::vector<std::vector<float>> sort(std::vector<std::vector<float>>& _list) {
  std::vector<std::vector<float>> list(_list);
  int list_num = list.size();

  for (size_t i = 0; i < list_num; i++) {
    int val_num = list[i].size();
    float conf_max = list[i][val_num - 1];
    int index = i;
    for (size_t j = i + 1; j < list_num; j++) {
      val_num = list[j].size();
      float conf_val = list[j][val_num - 1];
      if (conf_val > conf_max) {
        conf_max = conf_val;
        index = j;
      }
    }
    if (index != i) {
      std::swap(list[i], list[index]);
    }
  }

  list.swap(_list);
  return _list;
}

float overlap_similarity(std::vector<float> r1, std::vector<float> r2) {
  float xmin1 = r1[1];
  float ymin1 = r1[0];
  float xmax1 = r1[3];
  float ymax1 = r1[2];

  float w1 = xmax1 - xmin1;
  float h1 = ymax1 - ymin1;

  float xmin2 = r2[1];
  float ymin2 = r2[0];
  float xmax2 = r2[3];
  float ymax2 = r2[2];

  float w2 = xmax2 - xmin2;
  float h2 = ymax2 - ymin2;

  float overlapW = std::min(xmax1, xmax2) - std::max(xmin1, xmin2);
  float overlapH = std::min(ymax1, ymax2) - std::max(ymin1, ymin2);

  return (overlapW * overlapH) /
         ((w1 * h1) + (w2 * h2) - (overlapW * overlapH));
}

inline void op_divide(float& a, float b) { a = a / b; }
inline void op_ride(float& a, float b) { a = a * b; }
inline float op_add(float a, float b) { return a + b; }
std::vector<std::vector<float>> WeightedNonMaxSuppression(
    std::vector<std::vector<float>> _list) {
  std::vector<std::vector<float>> res;
  std::vector<std::vector<float>> list(_list);
  int list_num = list.size();
  for (size_t i = 0; i < list_num; i++) {
    float conf = list[i][list[i].size() - 1];
    if (conf < 0) {
      continue;
    }

    std::vector<std::vector<float>> temp_face;
    temp_face.push_back(_list[i]);
    for (size_t j = i + 1; j < list_num; j++) {
      if (list[j][list[j].size() - 1] < 0) {
        continue;
      }
      float iou_val = overlap_similarity(list[i], list[j]);
      if (iou_val > 0.3) {
        list[j][list[j].size() - 1] = -1;
        temp_face.push_back(_list[j]);
      }
    }

    if (temp_face.size() > 0) {
      for (size_t j = 0; j < temp_face.size(); j++)
        for (size_t k = 0; k < temp_face[j].size() - 1; k++)
          op_ride(temp_face[j][k], temp_face[j][temp_face[j].size() - 1]);

      std::vector<float> temp_total_val(temp_face[0]);
      for (size_t j = 1; j < temp_face.size(); j++)
        std::transform(temp_face[j].begin(), temp_face[j].end(),
                       temp_total_val.begin(), temp_total_val.begin(), op_add);

      for (size_t j = 0; j < temp_total_val.size() - 1; j++)
        op_divide(temp_total_val[j], temp_total_val[temp_total_val.size() - 1]);

      temp_total_val[temp_total_val.size() - 1] /= temp_face.size();
      res.push_back(temp_total_val);
    }
  }
  return res;
}

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
//   calculator: "PostProcessCalculator"
//   input_stream: "vector<unsigned char*>"
//   output_stream: "None"
// }

class PostProcessCalculator : public CalculatorBase {
 private:
  int mNumber{0};  //!< The number to classify
  std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
  SampleUniquePtr<nvinfer1::IExecutionContext> context;
  std::vector<void*> mDeviceBindings;
  std::vector<int> output_nbsize;
  void* mBuffer;
  void* mBuffer_out;
  bool bind_out_buffer;
  cv::VideoWriter writer;

 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<std::vector<unsigned char*>>();
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) {
    init_anchor();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) {
    static int frame_count = 0;
    auto& input = cc->Inputs().Index(0).Get<std::vector<unsigned char*>>();
    cv::Mat raw_image(*(cv::Mat*)input[2]);
    verifyOutput((float*)input[1], (float*)input[0], raw_image);
    return absl::OkStatus();
  }

  // out1: 1*896*1   classificator
  // out2: 1*896*16  regressors
  bool verifyOutput(float* classificator, float* regressors, cv::Mat& src) {
    int x_scale = 128;
    int y_scale = 128;
    int w_scale = 128;
    int h_scale = 128;
    const int outputSize = 896;  // blob0
    const int outputSize2 = 1;
    const int output2Size = 896;  // blob1
    const int output2Size2 = 16;
    float* out1 = classificator;
    float* out2 = regressors;
    float val{0.0f};
    int idx{0};

#if 0
    for (int i = 0; i < output2Size; i++)
    {
        //output[i] = exp(output[i]);
        //sum +=
        printf("output[%d]=%f\r\n",i,out1[i]);

    }
#endif
    std::vector<Object> objects;
    //--------------获取定位
    // out1: 1*896*1   classificator
    // out2: 1*896*16  regressors
    std::vector<std::vector<float>> det_list;
    for (size_t h = 0; h < outputSize; h++) {
      for (size_t w = 0; w < outputSize2; w++) {
        if (sigmoid(out1[h * outputSize2 + w]) > 0.65) {
          std::vector<float> val;

          float x_center =
              out2[h * output2Size2 + 0] / x_scale * anchors[h][2] +
              anchors[h][0];
          float y_center =
              out2[h * output2Size2 + 1] / y_scale * anchors[h][3] +
              anchors[h][1];
          float bw = out2[h * output2Size2 + 2] / w_scale * anchors[h][2];
          float bh = out2[h * output2Size2 + 3] / h_scale * anchors[h][3];

          val.push_back(y_center - bh / 2.0);
          val.push_back(x_center - bw / 2.0);
          val.push_back(y_center + bh / 2.0);
          val.push_back(x_center + bw / 2.0);

          for (size_t k = 0; k < 6; k++) {
            int offset = 4 + k * 2;
            float keypoint_x =
                out2[h * output2Size2 + offset] / x_scale * anchors[h][2] +
                anchors[h][0];
            float keypoint_y =
                out2[h * output2Size2 + offset + 1] / y_scale * anchors[h][3] +
                anchors[h][1];
            val.push_back(keypoint_x);
            val.push_back(keypoint_y);
          }
          val.push_back(sigmoid(out1[h * outputSize2 + w]));
          det_list.push_back(val);
        }
      }
    }

    sort(det_list);
    std::vector<std::vector<float>> res_face =
        WeightedNonMaxSuppression(det_list);
    for (size_t i = 0; i < res_face.size(); i++) {
      Object face;
      face.rect.x = res_face[i][1];
      face.rect.y = res_face[i][0];
      face.rect.width = res_face[i][3] - res_face[i][1];
      face.rect.height = res_face[i][2] - res_face[i][0];
      face.prob = res_face[i][16];
      for (size_t k = 0; k < 6; k++) {
        int offset = 4 + k * 2;
        cv::Point2f kp;

        kp.x = res_face[i][offset];
        kp.y = res_face[i][offset + 1];
        face.landmarks.push_back(kp);
      }
      objects.push_back(face);
    }
    // cv::Mat src = cv::imread("image.jpg");
    for (size_t i = 0; i < objects.size(); i++) {
      int x1 = objects[i].rect.x * src.cols;
      int y1 = objects[i].rect.y * src.rows;
      int width1 = objects[i].rect.width * src.cols;
      int height1 = objects[i].rect.height * src.rows;
      // printf("%f %f %f %f\n", objects[i].rect.x, objects[i].rect.y,
      // objects[i].rect.width, objects[i].rect.height);
      cv::rectangle(src, cv::Rect(cv::Point(x1, y1), cv::Size(width1, height1)),
                    cv::Scalar(0, 0, 255), 3);

      char text[256];
      sprintf_s(text, "%.1f%%", objects[i].prob * 100);

      int baseLine = 0;
      cv::Size label_size =
          cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
      cv::putText(src, text, cv::Point(x1, y1 - label_size.height),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));

      for (size_t k = 0; k < 6; k++) {
        cv::Point kp;
        kp.x = objects[i].landmarks[k].x * src.cols;
        kp.y = objects[i].landmarks[k].y * src.rows;
        cv::circle(src, cv::Point(kp.x, kp.y), 3, cv::Scalar(255, 0, 0), 2);
      }
    }
#if 0
    static int saved_frame=0;
    std::string res_name="./trt_detectresult/" + std::to_string(saved_frame) + ".jpg";
	cv::imwrite(res_name, src);
    saved_frame++;
#endif
    if (!writer.isOpened()) {
      LOG(INFO) << "Prepare video writer.";
      writer.open("trt_infer.mp4",
                  mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                  25, src.size());
      // RET_CHECK(writer.isOpened());
    }
    cv::cvtColor(src, src, cv::COLOR_RGB2BGR);
    writer.write(src);
    return true;
  }
};

REGISTER_CALCULATOR(PostProcessCalculator);
}  // namespace mediapipe
