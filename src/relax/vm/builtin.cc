/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relax/vm/builtin.cc
 * \brief 
 */
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

namespace tvm {
namespace relax {
namespace vm {

using tvm::runtime::NDArray;

TVM_REGISTER_GLOBAL("vm.builtin.shape_of").set_body_typed(
[](NDArray arr) {
  return arr.Shape();
});

TVM_REGISTER_GLOBAL("vm.builtin.alloc_heap").set_body_typed(
[](int64_t size) {
  return NDArray::Empty(ShapeTuple({size}),
                        DLDataType{kDLInt, 64, 1},
                        DLDevice{kDLCPU, 0});
});

TVM_REGISTER_GLOBAL("vm.builtin.match_shape").set_body(
[] (runtime::TVMArgs args, runtime::TVMRetValue* rv) {
  ShapeTuple shape = args[0];
  NDArray heap = args[1];
  int64_t* heap_data = reinterpret_cast<int64_t*>(heap.ToDLPack()->dl_tensor.data);
  for (int i = 2; i < args.size(); ++i) {
    int64_t heap_idx = args[i];
    heap_data[heap_idx] = shape[i - 2];
  }
});

TVM_REGISTER_GLOBAL("vm.builtin.make_shape").set_body(
[](runtime::TVMArgs args, runtime::TVMRetValue* rv) {
  NDArray heap = args[0];
  int64_t* heap_data = reinterpret_cast<int64_t*>(heap.ToDLPack()->dl_tensor.data);
  std::vector<int64_t> shape;
  for (int i = 1; i < args.size(); ++i) {
    int64_t heap_idx = args[i];
    shape.push_back(heap_data[heap_idx]);
  }
  return ShapeTuple(shape);
});

}  // namespace vm
}  // namespace relax
}  // namespace tvm
