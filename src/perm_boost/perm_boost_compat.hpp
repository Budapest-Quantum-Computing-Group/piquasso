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

/*
 * Portions of this file are based on work by Bence Soóki-Tóth, used with
 * permission and originally made available under the MIT License.
 *
 * Bence Soóki-Tóth. "Efficient calculation of permanent function gradients
 * in photonic quantum computing simulations", Eötvös Loránd University, 2025.
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

#pragma once

#include "xla/ffi/api/ffi.h"
#include <type_traits>
#include <cstdint>

namespace pq_compat {

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// buffer_data<dtype>(buf)  →  NativeType<dtype>*
// ---------------------------------------------------------------------------

// Primary template: old API — buf.data is a public typed pointer.
template <ffi::DataType dtype, typename = void>
struct BufferDataAccess {
    static ffi::NativeType<dtype>* get(const ffi::Buffer<dtype>& buf) {
        return buf.data;
    }
};

// Specialisation: new API — typed_data() method exists.
template <ffi::DataType dtype>
struct BufferDataAccess<dtype,
    std::void_t<decltype(std::declval<ffi::Buffer<dtype>>().typed_data())>> {
    static ffi::NativeType<dtype>* get(const ffi::Buffer<dtype>& buf) {
        return buf.typed_data();
    }
};

template <ffi::DataType dtype>
inline ffi::NativeType<dtype>* buffer_data(const ffi::Buffer<dtype>& buf) {
    return BufferDataAccess<dtype>::get(buf);
}

// Convenience for ResultBuffer: Result<Buffer<dtype>> has operator*().
template <ffi::DataType dtype>
inline ffi::NativeType<dtype>* result_buffer_data(ffi::ResultBuffer<dtype>& r) {
    return buffer_data<dtype>(*r);
}

// ---------------------------------------------------------------------------
// buffer_dimensions(buf)  →  span-like over const int64_t
// ---------------------------------------------------------------------------

// Primary template: old API — buf.dimensions is a public Span<const int64_t>.
template <ffi::DataType dtype, typename = void>
struct BufferDimsAccess {
    static auto get(const ffi::Buffer<dtype>& buf)
        -> decltype(buf.dimensions) {
        return buf.dimensions;
    }
};

// Specialisation: new API — dimensions() method exists.
template <ffi::DataType dtype>
struct BufferDimsAccess<dtype,
    std::void_t<decltype(std::declval<ffi::Buffer<dtype>>().dimensions())>> {
    static auto get(const ffi::Buffer<dtype>& buf)
        -> decltype(buf.dimensions()) {
        return buf.dimensions();
    }
};

template <ffi::DataType dtype>
inline auto buffer_dimensions(const ffi::Buffer<dtype>& buf) {
    return BufferDimsAccess<dtype>::get(buf);
}

// ---------------------------------------------------------------------------
// buffer_element_count(buf)  →  int64_t
// ---------------------------------------------------------------------------

// Primary template: old API — no element_count(); compute from dimensions.
template <ffi::DataType dtype, typename = void>
struct BufferElementCountAccess {
    static int64_t get(const ffi::Buffer<dtype>& buf) {
        int64_t count = 1;
        for (int64_t d : buf.dimensions) { count *= d; }
        return count;
    }
};

// Specialisation: new API — element_count() method exists.
template <ffi::DataType dtype>
struct BufferElementCountAccess<dtype,
    std::void_t<decltype(static_cast<int64_t>(
        std::declval<ffi::Buffer<dtype>>().element_count()))>> {
    static int64_t get(const ffi::Buffer<dtype>& buf) {
        return static_cast<int64_t>(buf.element_count());
    }
};

template <ffi::DataType dtype>
inline int64_t buffer_element_count(const ffi::Buffer<dtype>& buf) {
    return BufferElementCountAccess<dtype>::get(buf);
}

} // namespace pq_compat
