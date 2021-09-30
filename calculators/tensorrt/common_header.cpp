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

#include "common_header.h"

int save_array2file(void* array2save, int size) {
  FILE* dst_file = fopen("data_from_texturebuffer.raw", "wb+");
  if (dst_file) {
    int w_len = fwrite((char*)array2save, 1, size, dst_file);
    fflush(dst_file);
    fclose(dst_file);
    return (w_len == size);
  } else {
    printf("cannot create file");
    return 0;
  }
}
