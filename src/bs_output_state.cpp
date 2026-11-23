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
 * The output state is built particle by particle (SLOS): the `n` input particles
 * are sent through the interferometer one at a time, in the order of their input
 * modes.
 *
 * Descending lexicographic order
 *     The order of occupations throughout this file, the same as in `partitions`
 *     of `piquasso/_math/combinatorics.py`. For 3 modes and 2 particles:
 *     (2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2).
 *
 * Layer k
 *     The state after the first `k` input particles have been added: a vector of
 *     amplitudes of occupations with `k` particles. The position of the amplitude
 *     of an occupation in this vector is its index. Layer 0 is the vacuum, layer
 *     `n` is the result. Each layer is computed from the previous one, so only
 *     two of them are kept in memory.
 *
 * Full layer
 *     A layer of all `C(k + d - 1, d - 1)` occupations with `k` particles, i.e.
 *     the whole k-particle Fock subspace. Without postselection, every layer is
 *     full.
 *
 * Rank
 *     The index of an occupation in the full layer of its particle number.
 *     `rank_with_suffix` computes it from the suffix sums
 *     `S_i = x_i + ... + x_{d-1}`, so the occupations of a layer are never
 *     stored.
 *
 * Postselected layer
 *     A layer of the admissible occupations only, i.e. of those which can still
 *     grow into a postselected output (see `Constraints`), in the same order.
 *     Since the other occupations are skipped, indices differ from ranks.
 *
 * Layer index
 *     Maps the rank of an occupation to its index in a postselected layer, or to
 *     `NOT_IN_BASIS` if the occupation is not admissible (see `LayerIndex`).
 */

#include "bs_output_state.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <numeric>

namespace
{

using Occupation = std::vector<std::int32_t>;

constexpr std::int64_t NOT_IN_BASIS = -1;

/**
 * Table of binomial coefficients `C(a, b)` for `a < top` and `b < width`, used
 * for ranking.
 */
struct Binomials
{
    std::size_t width;
    std::vector<std::size_t> table;
};

Binomials make_binomials(std::size_t top, std::size_t width)
{
    Binomials binomials{width, std::vector<std::size_t>(top * width, 0)};
    auto &table = binomials.table;

    for (std::size_t a = 0; a < top; ++a)
    {
        table[a * width] = 1;

        // `C(a - 1, a) = 0` is read from the zero-initialized table.
        for (std::size_t b = 1; b < std::min(a + 1, width); ++b)
        {
            const auto left = table[(a - 1) * width + b - 1];
            const auto right = table[(a - 1) * width + b];

            table[a * width + b] = left + right;
        }
    }

    return binomials;
}

inline std::size_t binomial(const Binomials &binomials, std::size_t a, std::size_t b)
{
    return binomials.table[a * binomials.width + b];
}

/**
 * Computes the suffix sums `S_i` of `state` into `suffix`, and returns the rank
 * of `state`, which only depends on `S_i` for `i > 0`.
 */
std::size_t rank_with_suffix(
    const Occupation &state,
    std::vector<std::int32_t> &suffix,
    const Binomials &binomials
)
{
    const auto d = state.size();
    std::size_t rank = 0;

    suffix[d] = 0;

    // From the last mode to the first, so that each suffix sum extends the next
    // one. `i-- > 0` decrements after the test, so the body sees `i = d - 1, ...,
    // 0` without the unsigned `i` wrapping around. (Clearer forms of this loop
    // measured slower.)
    for (std::size_t i = d; i-- > 0;)
    {
        suffix[i] = suffix[i + 1] + state[i];

        if (i > 0)
            rank += binomial(binomials, suffix[i] + (d - 1 - i), d - i);
    }

    return rank;
}

/**
 * Change of the rank from `x + e_{i-1}` to `x + e_i`, `i > 0`, where `suffix_i`
 * is the suffix sum `S_i` of `x`. Only `S_i` changes, by one, so by Pascal's
 * identity, the rank grows by `C(S_i + d - 1 - i, d - 1 - i)`.
 */
inline std::size_t rank_increment(
    const Binomials &binomials, std::size_t d, std::size_t i, std::int32_t suffix_i
)
{
    const auto j = d - 1 - i;

    return binomial(binomials, suffix_i + j, j);
}

/**
 * Advances `state` to the next occupation in descending lexicographic order, as
 * `partitions` of `piquasso/_math/combinatorics.py`. `pivot` is the rightmost
 * nonzero mode before the last one. It becomes -1 at the last occupation, which
 * must not be advanced.
 */
void next_occupation(Occupation &state, std::ptrdiff_t &pivot)
{
    const auto last = static_cast<std::ptrdiff_t>(state.size()) - 1;

    const auto tail = state[last];

    state[last] = 0;
    --state[pivot];
    state[pivot + 1] = tail + 1;

    if (pivot + 1 < last)
    {
        ++pivot;
    }
    else if (state[pivot] == 0)
    {
        --pivot;

        while (pivot >= 0 && state[pivot] == 0)
            --pivot;
    }
}

/**
 * An occupation `x` is admissible if `x[j] <= target[j]` in every postselected
 * mode `j`, and the number of photons in the free (not postselected) modes is at
 * most `free_limit`, the number of free photons in the output.
 */
struct Constraints
{
    std::vector<std::int32_t> upper;
    std::vector<bool> is_free;
    std::int64_t free_limit;
};

Constraints make_constraints(
    std::size_t d,
    std::int64_t n,
    const std::vector<std::int32_t> &modes,
    const std::vector<std::int32_t> &photons
)
{
    Constraints constraints{
        std::vector<std::int32_t>(d, std::numeric_limits<std::int32_t>::max()),
        std::vector<bool>(d, true),
        n,
    };

    for (std::size_t i = 0; i < modes.size(); ++i)
    {
        const auto mode = static_cast<std::size_t>(modes[i]);

        constraints.upper[mode] = photons[i];
        constraints.is_free[mode] = false;
        constraints.free_limit -= photons[i];
    }

    return constraints;
}

/**
 * The recursion of `for_each_admissible`: fills the modes of `state` from `mode`
 * on with the `remaining` particles, where the modes before `mode` hold
 * `free_used` photons in free modes.
 */
template <typename TCallback>
void for_each_admissible_from(
    const Constraints &constraints,
    std::size_t mode,
    std::int32_t remaining,
    std::int64_t free_used,
    Occupation &state,
    TCallback &callback
)
{
    const bool is_last = mode + 1 == state.size();

    auto max_value = std::min<std::int64_t>(remaining, constraints.upper[mode]);

    if (constraints.is_free[mode])
        max_value = std::min(max_value, constraints.free_limit - free_used);

    const std::int64_t min_value = is_last ? remaining : 0;

    for (auto value = max_value; value >= min_value; --value)
    {
        state[mode] = static_cast<std::int32_t>(value);

        if (is_last)
            callback(state);
        else
            for_each_admissible_from(
                constraints,
                mode + 1,
                remaining - static_cast<std::int32_t>(value),
                free_used + (constraints.is_free[mode] ? value : 0),
                state,
                callback
            );
    }

    state[mode] = 0;
}

/**
 * Calls `callback(state)` on every admissible occupation of `particles`
 * particles, in descending lexicographic order, without visiting any other
 * occupation.
 */
template <typename TCallback>
void for_each_admissible(
    const Constraints &constraints, std::int32_t particles, TCallback &&callback
)
{
    Occupation state(constraints.upper.size(), 0);

    for_each_admissible_from(constraints, 0, particles, 0, state, callback);
}

/**
 * Maps the rank of an admissible occupation to its index in a postselected
 * layer. The indices follow the order of `for_each_admissible`. If `is_dense`,
 * the index is `values[rank]`, otherwise `values[slot]` with `keys[slot] == rank`
 * in an open-addressing hash table.
 */
struct LayerIndex
{
    std::size_t size;
    bool is_dense;
    std::size_t mask;
    std::vector<std::size_t> keys;
    std::vector<std::int64_t> values;
};

constexpr std::size_t DENSE_LIMIT = std::size_t(1) << 22;
constexpr std::size_t EMPTY = std::numeric_limits<std::size_t>::max();

inline std::size_t scramble(std::size_t value)
{
    value ^= value >> 32;
    value *= 0x9E3779B97F4A7C15ULL;

    return value >> 20;
}

/**
 * Builds the index of the postselected layer of `particles` particles. A dense
 * array is used if the full layer is small or not much larger than the
 * postselected one, otherwise a hash table.
 */
LayerIndex make_layer_index(
    const Constraints &constraints,
    std::size_t d,
    std::int32_t particles,
    const Binomials &binomials
)
{
    const auto full_size = binomial(binomials, particles + d - 1, d - 1);

    std::vector<std::size_t> ranks;
    std::vector<std::int32_t> suffix(d + 1, 0);

    auto collect = [&](const Occupation &admissible)
    {
        ranks.push_back(rank_with_suffix(admissible, suffix, binomials));
    };

    for_each_admissible(constraints, particles, collect);

    LayerIndex index{ranks.size(), true, 0, {}, {}};

    if (full_size <= DENSE_LIMIT || full_size <= 16 * index.size)
    {
        index.values.assign(full_size, NOT_IN_BASIS);

        for (std::size_t idx = 0; idx < index.size; ++idx)
            index.values[ranks[idx]] = static_cast<std::int64_t>(idx);

        return index;
    }

    std::size_t capacity = 4;

    while (capacity < 2 * index.size)
        capacity <<= 1;

    index.is_dense = false;
    index.mask = capacity - 1;
    index.keys.assign(capacity, EMPTY);
    index.values.assign(capacity, NOT_IN_BASIS);

    for (std::size_t idx = 0; idx < index.size; ++idx)
    {
        auto slot = scramble(ranks[idx]) & index.mask;

        while (index.keys[slot] != EMPTY)
            slot = (slot + 1) & index.mask;

        index.keys[slot] = ranks[idx];
        index.values[slot] = static_cast<std::int64_t>(idx);
    }

    return index;
}

/** Returns the index of the occupation of the given rank, or `NOT_IN_BASIS`. */
inline std::int64_t find(const LayerIndex &index, std::size_t rank)
{
    if (index.is_dense)
        return index.values[rank];

    auto slot = scramble(rank) & index.mask;

    while (index.keys[slot] != EMPTY)
    {
        if (index.keys[slot] == rank)
            return index.values[slot];

        slot = (slot + 1) & index.mask;
    }

    return NOT_IN_BASIS;
}

/**
 * Adds one particle, entering through `mode` of the input, to every occupation
 * of the previous layer, which has `particles` particles. `already_in_mode`
 * particles have been sent through `mode` before. The layers are full: they are
 * walked with `next_occupation`, and the index of an occupation is its rank.
 */
template <typename TScalar>
Vector<TScalar> add_particle_full(
    const Matrix<TScalar> &interferometer,
    const Vector<TScalar> &previous,
    const Binomials &binomials,
    const std::vector<typename TScalar::value_type> &sqrt_table,
    std::int32_t particles,
    std::size_t mode,
    std::int32_t already_in_mode
)
{
    using TReal = typename TScalar::value_type;

    const auto d = interferometer.rows;
    const auto next_size = binomial(binomials, particles + d, d - 1);

    Vector<TScalar> next(next_size);
    std::fill(next.data, next.data + next_size, TScalar(0));

    const auto inverse_norm = TReal(1) / sqrt_table[already_in_mode + 1];

    Occupation state(d, 0);
    state[0] = particles;
    std::ptrdiff_t pivot = 0;

    std::vector<std::int32_t> suffix(d + 1, 0);

    for (std::size_t idx = 0; idx < previous.length; ++idx)
    {
        const auto weight = previous[idx] * inverse_norm;

        // The rank of `state + e_0` is the rank of `state`, and the rank of
        // `state + e_i` follows incrementally.
        auto rank = rank_with_suffix(state, suffix, binomials);

        for (std::size_t i = 0; i < d; ++i)
        {
            if (i > 0)
                rank += rank_increment(binomials, d, i, suffix[i]);

            next[rank] += sqrt_table[state[i] + 1] * interferometer(i, mode) * weight;
        }

        if (idx + 1 < previous.length)
            next_occupation(state, pivot);
    }

    return next;
}

/**
 * Same as `add_particle_full`, but only the admissible occupations are visited,
 * and the resulting occupations `x + e_i` are looked up in `next_index`, the
 * index of the next layer.
 */
template <typename TScalar>
Vector<TScalar> add_particle_postselected(
    const Matrix<TScalar> &interferometer,
    const Vector<TScalar> &previous,
    const Constraints &constraints,
    const LayerIndex &next_index,
    const Binomials &binomials,
    const std::vector<typename TScalar::value_type> &sqrt_table,
    std::int32_t particles,
    std::size_t mode,
    std::int32_t already_in_mode
)
{
    using TReal = typename TScalar::value_type;

    const auto d = interferometer.rows;

    Vector<TScalar> next(next_index.size);
    std::fill(next.data, next.data + next.length, TScalar(0));

    const auto inverse_norm = TReal(1) / sqrt_table[already_in_mode + 1];

    std::vector<std::int32_t> suffix(d + 1, 0);
    std::size_t idx = 0;

    auto update = [&](const Occupation &admissible)
    {
        const auto weight = previous[idx++] * inverse_norm;

        auto rank = rank_with_suffix(admissible, suffix, binomials);

        for (std::size_t i = 0; i < d; ++i)
        {
            if (i > 0)
                rank += rank_increment(binomials, d, i, suffix[i]);

            const auto found = find(next_index, rank);

            if (found != NOT_IN_BASIS)
                next[found] += sqrt_table[admissible[i] + 1] * interferometer(i, mode) * weight;
        }
    };

    for_each_admissible(constraints, particles, update);

    return next;
}

} // namespace

template <typename TScalar>
Vector<TScalar> calculate_BS_output_state(
    const Matrix<TScalar> &interferometer,
    const std::vector<std::int32_t> &initial_state,
    const std::vector<std::int32_t> &postselect_modes,
    const std::vector<std::int32_t> &postselect_photons
)
{
    using TReal = typename TScalar::value_type;

    const auto d = initial_state.size();

    if (d == 0)
    {
        Vector<TScalar> result(1);
        result[0] = TScalar(1);

        return result;
    }

    const auto n = std::accumulate(initial_state.begin(), initial_state.end(), std::int64_t{0});

    // `C(a, b)` for `a < n + d` and `b < d`.
    const auto binomials = make_binomials(n + d, d);

    // `sqrt(v)` for `v <= n`.
    std::vector<TReal> sqrt_table(n + 1);

    for (std::size_t v = 0; v < sqrt_table.size(); ++v)
        sqrt_table[v] = std::sqrt(static_cast<TReal>(v));

    const bool is_postselected = !postselect_modes.empty();
    const auto constraints = make_constraints(d, n, postselect_modes, postselect_photons);

    if (constraints.free_limit < 0)
        return Vector<TScalar>(0);

    // Only two consecutive layers are alive at any time. A deque is used, since
    // `Vector` assignment does not release the previous data, but its destructor
    // does.
    std::deque<Vector<TScalar>> layers;

    // Layer 0 is the vacuum, unless the vacuum is not admissible (more photons are
    // postselected than there are in the input, and some mode is free). Then
    // layer 0 is empty, which matters for `n = 0`, when it is the result.
    layers.emplace_back(
        is_postselected ? make_layer_index(constraints, d, 0, binomials).size : 1
    );
    std::fill(layers[0].data, layers[0].data + layers[0].length, TScalar(0));

    if (layers[0].length > 0)
    {
        layers[0][0] = TScalar(1);
    }

    std::int32_t particles = 0;

    for (std::size_t mode = 0; mode < d; ++mode)
    {
        // `j` particles have already been sent through `mode`.
        for (std::int32_t j = 0; j < initial_state[mode]; ++j, ++particles)
        {
            if (is_postselected)
            {
                const auto next_index =
                    make_layer_index(constraints, d, particles + 1, binomials);

                layers.push_back(add_particle_postselected(
                    interferometer,
                    layers.back(),
                    constraints,
                    next_index,
                    binomials,
                    sqrt_table,
                    particles,
                    mode,
                    j
                ));
            }
            else
            {
                layers.push_back(add_particle_full(
                    interferometer, layers.back(), binomials, sqrt_table, particles, mode, j
                ));
            }

            layers.pop_front();
        }
    }

    return layers.back();
}

template Vector<std::complex<float>> calculate_BS_output_state(
    const Matrix<std::complex<float>> &,
    const std::vector<std::int32_t> &,
    const std::vector<std::int32_t> &,
    const std::vector<std::int32_t> &
);

template Vector<std::complex<double>> calculate_BS_output_state(
    const Matrix<std::complex<double>> &,
    const std::vector<std::int32_t> &,
    const std::vector<std::int32_t> &,
    const std::vector<std::int32_t> &
);
