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

#include <cstdio>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
// customer calculator

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"

class CustomerDataType {
 public:
  CustomerDataType(int i, float f, bool b, const std::string& str)
      : val_i(i), val_f(f), val_b(b), s_str(str) {}
  int val_i = 1;
  float val_f = 11.f;
  bool val_b = true;
  std::string s_str = "customer str.";
};

namespace mediapipe {

class MyStringProcessCalculator : public CalculatorBase {
 public:
  /*
  Calculator authors can specify the expected types of inputs and outputs of a
  calculator in GetContract(). When a graph is initialized, the framework calls
  a static method to verify if the packet types of the connected inputs and
  outputs match the information in this specification.
  */
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->Inputs().Index(1).Set<CustomerDataType>();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));

    if (cc->OutputSidePackets().NumEntries() != 0) {
      cc->OutputSidePackets().Index(0).SetSameAs(
          &cc->InputSidePackets().Index(0));
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    for (CollectionItemId id = cc->Inputs().BeginId();
         id < cc->Inputs().EndId(); ++id) {
      if (!cc->Inputs().Get(id).Header().IsEmpty()) {
        cc->Outputs().Get(id).SetHeader(cc->Inputs().Get(id).Header());
      }
    }
    if (cc->OutputSidePackets().NumEntries() != 0) {
      for (CollectionItemId id = cc->InputSidePackets().BeginId();
           id < cc->InputSidePackets().EndId(); ++id) {
        cc->OutputSidePackets().Get(id).Set(cc->InputSidePackets().Get(id));
      }
    }
    // Sets this packet timestamp offset for Packets going to all outputs.
    // If you only want to set the offset for a single output stream then
    // use OutputStream::SetOffset() directly.
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    if (cc->Inputs().NumEntries() == 0) {
      return tool::StatusStop();
    }

    // get node input data
    mediapipe::Packet _data0 = cc->Inputs().Index(0).Value();
    mediapipe::Packet _data1 = cc->Inputs().Index(1).Value();
    // not safety.
    char _tmp_buf[1024];
    ::memset(_tmp_buf, 0, 1024);
    snprintf(_tmp_buf, 1024, _data0.Get<std::string>().c_str(),
             _data1.Get<CustomerDataType>().val_i,
             _data1.Get<CustomerDataType>().val_f,
             _data1.Get<CustomerDataType>().val_b,
             _data1.Get<CustomerDataType>().s_str.c_str());
    std::string _out_data = _tmp_buf;
    cc->Outputs().Index(0).AddPacket(
        MakePacket<std::string>(_out_data).At(cc->InputTimestamp()));
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) final { return absl::OkStatus(); }
};

REGISTER_CALCULATOR(MyStringProcessCalculator);
}  // namespace mediapipe
