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
#include <chrono>
#include <cstdlib>
#include <memory>
#include <vector>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "common_header.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");
ABSL_FLAG(bool, remote_run, true,
          "FLAGS_remote_run. "
          "If not provided, default true");

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
// extern absl::Flag<bool> FLAGS_remote_run;

// test demo
namespace mediapipe {

absl::Status RunMyGraph() {
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  // Configures a simple graph, which concatenates 2 PassThroughCalculators.
  CalculatorGraphConfig config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
      input_stream: "input_video"

      node {
        calculator: "GpuBufToTrtBufCalculator"
        input_stream: "input_video"
        output_stream: "out_video_for_trt"
      }
      node {
        calculator: "PassThroughCalculator"
        input_stream: "out_video_for_trt"
        output_stream: "out_for_infer"
      }
      node {
        calculator: "TrtInferenceCalculator"
        input_stream: "out_for_infer"
        output_stream: "out"
      }
      node {
        calculator: "PostProcessCalculator"
        input_stream: "out"
      }
    )");
  LOG(INFO) << "parse graph cfg-str done ... ...";

  CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));
  LOG(INFO) << "init graph done ... ...";

  LOG(INFO) << "Initialize the GPU.";
  ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
  MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
  mediapipe::GlCalculatorHelper gpu_helper;
  gpu_helper.InitializeForTest(graph.GetGpuResources().get());

  ASSIGN_OR_RETURN(OutputStreamPoller poller,
                   graph.AddOutputStreamPoller("out"));

  LOG(INFO) << "Start running the calculator graph.";

  MP_RETURN_IF_ERROR(graph.StartRun({}));
  LOG(INFO) << "start run graph done ... ...";
  auto t1 = high_resolution_clock::now();

  LOG(INFO) << "--Initialize the camera or load the video: ["
            << absl::GetFlag(FLAGS_input_video_path) << "].";
  cv::VideoCapture capture;
  const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
  if (load_video) {
    LOG(INFO) << "+++++" << absl::GetFlag(FLAGS_input_video_path);
    capture.open(absl::GetFlag(FLAGS_input_video_path));
  } else {
    capture.open(0);
  }
  RET_CHECK(capture.isOpened());

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;
  while (grab_frames) {
    // Capture opencv camera or video frame.
    int INPUT_SIZE = 128;
    int OUTPUT_NUM = 960;
    float X_SCALE = 10.0;
    float Y_SCALE = 10.0;
    float H_SCALE = 5.0;
    float W_SCALE = 5.0;
    float score_threshold = 0.7f;
    float nms_threshold = 0.3f;
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
      if (!load_video) {
        LOG(INFO) << "Ignore empty frames from camera.";
        continue;
      }
      LOG(INFO) << "Empty frame, end of video reached.";
      break;
    }
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGBA);
    if (!load_video) {
      cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    }
#if SINGLE_PIC
    cv::Mat camera_frame_raw = cv::imread(
        "/media/houmo/490fd486-af35-4236-aea5-f16fde9df9b5/zhenjiang/mediapipe/"
        "image.jpg");
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    cv::Mat image;
    cv::resize(camera_frame, image, cv::Size(INPUT_SIZE, INPUT_SIZE));
    image.convertTo(image, CV_32FC3);
    image = (image * 2 / 255.0f) - 1;
#endif

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGBA, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Prepare and add graph input packet.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(
        gpu_helper.RunInGlContext([&input_frame, &frame_timestamp_us, &graph,
                                   &gpu_helper]() -> absl::Status {
          // Convert ImageFrame to GpuBuffer.
          auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
          auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
          glFlush();
          texture.Release();
          // Send GPU image packet into the graph.
          MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
              kInputStream, mediapipe::Adopt(gpu_frame.release())
                                .At(mediapipe::Timestamp(frame_timestamp_us))));
          return absl::OkStatus();
        }));

    mediapipe::Packet packet;
    if (!poller.Next(&packet)) {
      LOG(ERROR) << "poller out packets error";
    }
    // auto& gpu_frame = packet.Get<std::vector<unsigned char*>>();
    LOG(INFO) << "Continue-----";
  }

  // Close the input stream.
  MP_RETURN_IF_ERROR(graph.CloseInputStream("input_video"));
  LOG(INFO) << "RunGraph Done";
  auto t2 = high_resolution_clock::now();
  duration<double, std::milli> ms_double = t2 - t1;
  LOG(INFO) << "Exclude Init Time Only RunGraph Time::" << ms_double.count()
            << "ms";
  return graph.WaitUntilDone();
}

}  // namespace mediapipe

int main(int argc, char** argv) {
  LOG(INFO) << "glog init success ... ...";
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  auto t1 = high_resolution_clock::now();
  static int loop_count = 0;
  if (1)  // change to while 1 for stress testing
  {
    absl::Status run_status = mediapipe::RunMyGraph();
    if (!run_status.ok()) {
      LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    } else {
      auto t2 = high_resolution_clock::now();
      duration<double, std::milli> ms_double = t2 - t1;
      LOG(INFO) << "TRT RunGraph Time::" << ms_double.count() << "ms";
    }
    LOG(INFO) << "=================Loop :" << loop_count;
    loop_count++;
  }
  google::ShutdownGoogleLogging();
  return 0;
}
