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

#ifndef PIQUASSO_POLYNOMIAL_HPP
#define PIQUASSO_POLYNOMIAL_HPP

#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>

#if defined(_MSC_VER)
#define PIQUASSO_RESTRICT __restrict
#else
#define PIQUASSO_RESTRICT __restrict__
#endif

// Arithmetic of polynomials in t with complex coefficients, modulo t^length,
// i.e., in the ring C[t] / (t^length). The real and imaginary parts of the
// coefficients are stored separately.

namespace piquasso::polynomial {

// Every polynomial is preceded by kPadding zero coefficients, so that negative
// indices read zeros.
constexpr std::size_t kPadding = 4;

struct Polynomial {
    double *re;
    double *im;
};

struct ConstPolynomial {
    const double *re;
    const double *im;

    ConstPolynomial(const double *real, const double *imag)
        : re(real), im(imag) {}
    ConstPolynomial(Polynomial polynomial)
        : re(polynomial.re), im(polynomial.im) {}
};

class PolynomialArray {
public:
    PolynomialArray() = default;

    PolynomialArray(std::size_t count, std::size_t cap)
        : cap_(cap), data_(count * stride(cap), 0.0) {}

    Polynomial operator[](std::size_t index) {
        double *base = data_.data() + index * stride(cap_);
        return {base + kPadding, base + cap_ + 2 * kPadding};
    }

    ConstPolynomial operator[](std::size_t index) const {
        const double *base = data_.data() + index * stride(cap_);
        return {base + kPadding, base + cap_ + 2 * kPadding};
    }

private:
    static std::size_t stride(std::size_t cap) { return 2 * (cap + kPadding); }

    std::size_t cap_ = 0;
    std::vector<double> data_;
};

inline std::complex<double> coefficient(ConstPolynomial p, std::size_t k) {
    return {p.re[k], p.im[k]};
}

inline void set_coefficient(
    Polynomial p,
    std::size_t k,
    const std::complex<double> &value
) {
    p.re[k] = value.real();
    p.im[k] = value.imag();
}

// A view of f whose coefficient j is the coefficient j + k of f.
inline Polynomial from_degree(Polynomial f, std::size_t k) {
    return {f.re + k, f.im + k};
}

inline void set_zero(Polynomial p, std::size_t length) {
    std::fill(p.re, p.re + length, 0.0);
    std::fill(p.im, p.im + length, 0.0);
}

inline void copy(
    Polynomial destination,
    ConstPolynomial source,
    std::size_t length
) {
    std::copy(source.re, source.re + length, destination.re);
    std::copy(source.im, source.im + length, destination.im);
}

// out = x + y modulo t^length
inline void add(
    Polynomial out,
    ConstPolynomial x,
    ConstPolynomial y,
    std::size_t length
) {
    for (std::size_t k = 0; k < length; ++k) {
        out.re[k] = x.re[k] + y.re[k];
        out.im[k] = x.im[k] + y.im[k];
    }
}

// out = a x + b t y modulo t^length
inline void combine(
    Polynomial out,
    double a,
    ConstPolynomial x,
    double b,
    ConstPolynomial y,
    std::size_t length
) {
    for (std::size_t k = 0; k < length; ++k) {
        out.re[k] = a * x.re[k];
        out.im[k] = a * x.im[k];
    }
    for (std::size_t k = 1; k < length; ++k) {
        out.re[k] += b * y.re[k - 1];
        out.im[k] += b * y.im[k - 1];
    }
}

// out = f' modulo t^length
inline void derivative(Polynomial out, ConstPolynomial f, std::size_t length) {
    for (std::size_t k = 0; k < length; ++k) {
        out.re[k] = static_cast<double>(k + 1) * f.re[k + 1];
        out.im[k] = static_cast<double>(k + 1) * f.im[k + 1];
    }
}

// The kernels of the products below.
namespace detail {

// d[j] += x y[j]
inline void axpy(
    double *PIQUASSO_RESTRICT dr,
    double *PIQUASSO_RESTRICT di,
    const double *PIQUASSO_RESTRICT yr,
    const double *PIQUASSO_RESTRICT yi,
    double xr,
    double xi,
    std::size_t count
) {
    for (std::size_t j = 0; j < count; ++j) {
        dr[j] += xr * yr[j] - xi * yi[j];
        di[j] += xr * yi[j] + xi * yr[j];
    }
}

// d[j] += sum_{u<4} x_u y[j-u], with padded y
inline void axpy4(
    double *PIQUASSO_RESTRICT dr,
    double *PIQUASSO_RESTRICT di,
    const double *PIQUASSO_RESTRICT yr,
    const double *PIQUASSO_RESTRICT yi,
    const double *x,
    std::size_t count
) {
    const double *yr1 = yr - 1;
    const double *yi1 = yi - 1;
    const double *yr2 = yr - 2;
    const double *yi2 = yi - 2;
    const double *yr3 = yr - 3;
    const double *yi3 = yi - 3;
    const double x0r = x[0], x0i = x[1], x1r = x[2], x1i = x[3];
    const double x2r = x[4], x2i = x[5], x3r = x[6], x3i = x[7];
    for (std::size_t j = 0; j < count; ++j) {
        dr[j] += x0r * yr[j] - x0i * yi[j] + x1r * yr1[j] - x1i * yi1[j]
            + x2r * yr2[j] - x2i * yi2[j] + x3r * yr3[j] - x3i * yi3[j];
        di[j] += x0r * yi[j] + x0i * yr[j] + x1r * yi1[j] + x1i * yr1[j]
            + x2r * yi2[j] + x2i * yr2[j] + x3r * yi3[j] + x3i * yr3[j];
    }
}

} // namespace detail

// out += factor t^shift x y modulo t^length, with padded y
inline void mul_add(
    Polynomial out,
    ConstPolynomial x,
    ConstPolynomial y,
    std::size_t length,
    std::size_t shift = 0,
    double factor = 1.0
) {
    if (length <= shift)
        return;

    const std::size_t m = length - shift;
    double *dr = out.re + shift;
    double *di = out.im + shift;
    std::size_t i = 0;
    for (; i + 4 <= m; i += 4) {
        double scaled[8];
        for (std::size_t u = 0; u < 4; ++u) {
            scaled[2 * u] = factor * x.re[i + u];
            scaled[2 * u + 1] = factor * x.im[i + u];
        }
        detail::axpy4(dr + i, di + i, y.re, y.im, scaled, m - i);
    }
    for (; i < m; ++i) {
        detail::axpy(
            dr + i, di + i, y.re, y.im, factor * x.re[i], factor * x.im[i],
            m - i
        );
    }
}

// out = x y modulo t^length, with padded y
inline void product(
    Polynomial out,
    ConstPolynomial x,
    ConstPolynomial y,
    std::size_t length
) {
    set_zero(out, length);
    mul_add(out, x, y, length);
}

// out = x0 y0 + x1 y1 modulo t^length, with padded y0 and y1
inline void product_sum(
    Polynomial out,
    ConstPolynomial x0,
    ConstPolynomial y0,
    ConstPolynomial x1,
    ConstPolynomial y1,
    std::size_t length
) {
    set_zero(out, length);
    mul_add(out, x0, y0, length);
    mul_add(out, x1, y1, length);
}

// out = x0 y0 - x1 y1 modulo t^length, with padded y0 and y1
inline void product_difference(
    Polynomial out,
    ConstPolynomial x0,
    ConstPolynomial y0,
    ConstPolynomial x1,
    ConstPolynomial y1,
    std::size_t length
) {
    set_zero(out, length);
    mul_add(out, x0, y0, length);
    mul_add(out, x1, y1, length, 0, -1.0);
}

// inverse = 1/f modulo t^length, for f(0) = 1
inline void inverse_unit(
    Polynomial inverse,
    ConstPolynomial f,
    std::size_t length
) {
    if (length == 0)
        return;

    set_zero(inverse, length);
    inverse.re[0] = 1.0;
    for (std::size_t k = 0; k + 1 < length; ++k) {
        detail::axpy(
            inverse.re + k + 1, inverse.im + k + 1, f.re + 1, f.im + 1,
            -inverse.re[k], -inverse.im[k], length - 1 - k
        );
    }
}

// out = factor log(f) modulo t^(n+1), for f(0) = 1 and inverse = 1/f
inline void log_unit(
    Polynomial out,
    ConstPolynomial f,
    ConstPolynomial inverse,
    std::size_t n,
    double factor,
    Polynomial work
) {
    derivative(work, f, n);
    set_zero(out, n + 1);
    mul_add(out, work, inverse, n + 1, 1);
    for (std::size_t k = 1; k <= n; ++k) {
        const double scale = factor / static_cast<double>(k);
        out.re[k] *= scale;
        out.im[k] *= scale;
    }
}

// f = exp(h) modulo t^(n+1), for h(0) = 0
inline void exponential(
    Polynomial f,
    ConstPolynomial h,
    std::size_t n,
    Polynomial work
) {
    derivative(work, h, n);
    set_zero(f, n + 1);
    f.re[0] = 1.0;
    for (std::size_t k = 1; k <= n; ++k) {
        detail::axpy(
            f.re + k, f.im + k, work.re, work.im, f.re[k - 1], f.im[k - 1],
            n - k + 1
        );
        f.re[k] /= static_cast<double>(k);
        f.im[k] /= static_cast<double>(k);
    }
}

} // namespace piquasso::polynomial

#endif
