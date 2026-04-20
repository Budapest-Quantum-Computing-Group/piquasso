/*
 * Copyright 2021-2026 Budapest Quantum Computing Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Compatibility helpers bridging the old (jaxlib ≤ 0.4.30) and new (jaxlib ≥ 0.4.31)
// XLA FFI Buffer API.
//
// Old API: Buffer<T> is a plain struct with public members:
//   NativeType<T>*         data;
//   Span<const int64_t>    dimensions;
//
// New API: Buffer<T> is a class with private buf_ and accessor methods:
//   NativeType<T>*  typed_data() const;
//   Dimensions      dimensions() const;
//   size_t          element_count() const;
//
// Detection technique: two overloads accept (buf, int) and (buf, long).
// Calling with literal 0 (an int) prefers the first; if its decltype return
// is invalid (SFINAE), the compiler falls back to the second.

#pragma once

#include "xla/ffi/api/ffi.h"
#include <cstdint>

namespace pq_compat {

namespace ffi = xla::ffi;
namespace detail {

// typed_data() (new) vs .data member (old)
template<typename B> auto data(const B& b, int)  -> decltype(b.typed_data()) { return b.typed_data(); }
template<typename B> auto data(const B& b, long)                              { return b.data; }

// dimensions() method (new) vs .dimensions member (old)
template<typename B> auto dims(const B& b, int)  -> decltype(b.dimensions())  { return b.dimensions(); }
template<typename B> auto dims(const B& b, long)                               { return b.dimensions; }

// element_count() method (new) vs product of .dimensions (old)
template<typename B> auto   count(const B& b, int)  -> decltype(b.element_count()) { return static_cast<int64_t>(b.element_count()); }
template<typename B> int64_t count(const B& b, long) {
    int64_t n = 1;
    for (int64_t d : b.dimensions) n *= d;
    return n;
}

} // namespace detail

template<ffi::DataType dtype>
auto buffer_data(const ffi::Buffer<dtype>& buf) { return detail::data(buf, 0); }

template<ffi::DataType dtype>
auto buffer_dimensions(const ffi::Buffer<dtype>& buf) { return detail::dims(buf, 0); }

template<ffi::DataType dtype>
int64_t buffer_element_count(const ffi::Buffer<dtype>& buf) { return detail::count(buf, 0); }

// Result<Buffer<dtype>> has operator*() in both APIs.
template<ffi::DataType dtype>
auto result_buffer_data(ffi::ResultBuffer<dtype>& r) { return buffer_data<dtype>(*r); }

} // namespace pq_compat
