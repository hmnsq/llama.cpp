#ਅਨ੍ਤਰ੍ਭਾਵ "fwht.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ P 1.0f
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N -1.0f

// constant Hadamard matrix via Paley I construction
ਸਥਿਰ constexpr ਦਸ਼ਮਲਵ H12[12][12] = {
    { P, P, P, P, P, P, P, P, P, P, P, P },
    { P, N, P, N, P, P, P, N, N, N, P, N },
    { P, N, N, P, N, P, P, P, N, N, N, P },
    { P, P, N, N, P, N, P, P, P, N, N, N },
    { P, N, P, N, N, P, N, P, P, P, N, N },
    { P, N, N, P, N, N, P, N, P, P, P, N },
    { P, N, N, N, P, N, N, P, N, P, P, P },
    { P, P, N, N, N, P, N, N, P, N, P, P },
    { P, P, P, N, N, N, P, N, N, P, N, P },
    { P, P, P, P, N, N, N, P, N, N, P, N },
    { P, N, P, P, P, N, N, N, P, N, N, P },
    { P, P, N, P, P, P, N, N, N, P, N, N }
};

ਸਥਿਰ constexpr ਦਸ਼ਮਲਵ H20[20][20] = {
    { P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P },
    { P, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N },
    { P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P },
    { P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P },
    { P, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N },
    { P, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N },
    { P, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N },
    { P, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N },
    { P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P },
    { P, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N },
    { P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P },
    { P, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N },
    { P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P },
    { P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P },
    { P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P },
    { P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P },
    { P, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N },
    { P, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N },
    { P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P },
    { P, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N }
};

#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ P
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ N

ਸਾਂਚਾ <ਪੂਰਨ N, ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ fwht_kernel(ਅਚਲ T * __restrict__ src, ਦਸ਼ਮਲਵ * __restrict__ dst, ਅਚਲ int64_t n_ਪੰਕਤੀਆਂ,
                        ਅਚਲ ਦਸ਼ਮਲਵ scale, ਅਚਲ sycl::nd_ਵਸਤੂ<2> & ਵਸਤੂ) {
    ਅਚਲ sycl::sub_ਸਮੂਹ sg = ਵਸਤੂ.get_sub_ਸਮੂਹ();

    ਅਚਲ int64_t r = ਵਸਤੂ.get_global_id(0);
    ਜੇ (r >= n_ਪੰਕਤੀਆਂ) {
        ਵਾਪਸ;
    }

    src += r * N;
    dst += r * N;

    constexpr ਪੂਰਨ el_w = N / WARP_ਮਾਪ;
    static_assert(el_w >= 1 && N % WARP_ਮਾਪ == 0, "row must be a whole number of sub-group widths");

    ਦਸ਼ਮਲਵ     reg[el_w];
    ਅਚਲ ਪੂਰਨ lane = sg.get_local_ਰੇਖਿਕ_id();

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < el_w; ++i) {
        reg[i] = static_cast<ਦਸ਼ਮਲਵ>(src[i * WARP_ਮਾਪ + lane]) * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ h = 1; h < WARP_ਮਾਪ; h *= 2) {
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < el_w; ++j) {
            ਅਚਲ ਦਸ਼ਮਲਵ val  = reg[j];
            ਅਚਲ ਦਸ਼ਮਲਵ val2 = dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(sg, val, h, WARP_ਮਾਪ);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ h = WARP_ਮਾਪ; h < N; h *= 2) {
        ਅਚਲ ਪੂਰਨ ਕਦਮ = h / WARP_ਮਾਪ;
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < el_w; j += 2 * ਕਦਮ) {
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ k = 0; k < ਕਦਮ; ++k) {
                ਅਚਲ ਦਸ਼ਮਲਵ x = reg[j + k];
                ਅਚਲ ਦਸ਼ਮਲਵ y = reg[j + k + ਕਦਮ];

                reg[j + k]        = x + y;
                reg[j + k + ਕਦਮ] = x - y;
            }
        }
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < el_w; ++i) {
        dst[i * WARP_ਮਾਪ + lane] = reg[i];
    }
}

ਸਾਂਚਾ <ਪੂਰਨ N, ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ launch_fwht(ਅਚਲ T * src, ਦਸ਼ਮਲਵ * dst, ਅਚਲ int64_t n_ਪੰਕਤੀਆਂ, ਅਚਲ ਦਸ਼ਮਲਵ scale,
                        dpct::ਕਤਾਰ_ptr stream) {
    constexpr ਪੂਰਨ ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਖੰਡ = 4;

    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (n_ਪੰਕਤੀਆਂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਖੰਡ - 1) / ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਖੰਡ;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ਅਚਲ sycl::range<2> global(ਅੰਕ_ਖੰਡਾਂ * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਖੰਡ, WARP_ਮਾਪ);
    ਅਚਲ sycl::range<2> local(ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਖੰਡ, WARP_ਮਾਪ);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_ਵਸਤੂ<2> ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             fwht_kernel<N, T>(src, dst, n_ਪੰਕਤੀਆਂ, scale, ਵਸਤੂ);
                         });
}

// Wide blocks: one row per work-group instead of per sub-group, so each work-item
// keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
// still shuffle; those up to NT go through work-group local memory; the rest stay
// in registers.
ਸਾਂਚਾ <ਪੂਰਨ N, ਪੂਰਨ NT, ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ fwht_kernel_wide(ਅਚਲ T * __restrict__ src,
                             ਦਸ਼ਮਲਵ * __restrict__ dst,
                             ਅਚਲ int64_t            n_ਪੰਕਤੀਆਂ,
                             ਅਚਲ ਦਸ਼ਮਲਵ              scale,
                             ਅਚਲ sycl::nd_ਵਸਤੂ<2> & ਵਸਤੂ,
                             ਦਸ਼ਮਲਵ *                  smem) {
    ਅਚਲ int64_t r = ਵਸਤੂ.get_global_id(0);
    ਜੇ (r >= n_ਪੰਕਤੀਆਂ) {
        ਵਾਪਸ;
    }

    src += r * N;
    dst += r * N;

    constexpr ਪੂਰਨ el_w = N / NT;
    static_assert(el_w >= 1 && N % NT == 0, "row must be a whole number of work-group widths");

    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ.get_local_id(1);

    ਦਸ਼ਮਲਵ reg[el_w];
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < el_w; ++i) {
        reg[i] = static_cast<ਦਸ਼ਮਲਵ>(src[i * NT + tid]) * scale;
    }

    ਅਚਲ sycl::sub_ਸਮੂਹ sg   = ਵਸਤੂ.get_sub_ਸਮੂਹ();
    ਅਚਲ ਪੂਰਨ             lane = sg.get_local_ਰੇਖਿਕ_id();

    // Butterflies inside the sub-group, same pattern as the narrow kernel.
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ h = 1; h < WARP_ਮਾਪ; h *= 2) {
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < el_w; ++j) {
            ਅਚਲ ਦਸ਼ਮਲਵ val  = reg[j];
            ਅਚਲ ਦਸ਼ਮਲਵ val2 = dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(sg, val, h, WARP_ਮਾਪ);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies from the sub-group width up to NT: the partner lane is outside
    // this sub-group, so it goes through work-group local memory instead of a shuffle.
    ਲਈ (ਪੂਰਨ h = WARP_ਮਾਪ; h < NT; h *= 2) {
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < el_w; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        ਵਸਤੂ.barrier(sycl::access::fence_space::local_space);
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < el_w; ++j) {
            ਅਚਲ ਦਸ਼ਮਲਵ val  = reg[j];
            ਅਚਲ ਦਸ਼ਮਲਵ val2 = smem[j * NT + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        ਵਸਤੂ.barrier(sycl::access::fence_space::local_space);
    }

    // Butterflies across registers: h is a multiple of NT, so the partner of element
    // i*NT + tid lives in reg[i + h/NT] on the same work-item.
    ਲਈ (ਪੂਰਨ h = NT; h < N; h *= 2) {
        ਅਚਲ ਪੂਰਨ ਕਦਮ = h / NT;
        ਲਈ (ਪੂਰਨ j = 0; j < el_w; j += 2 * ਕਦਮ) {
            ਲਈ (ਪੂਰਨ k = 0; k < ਕਦਮ; ++k) {
                ਅਚਲ ਦਸ਼ਮਲਵ x = reg[j + k];
                ਅਚਲ ਦਸ਼ਮਲਵ y = reg[j + k + ਕਦਮ];

                reg[j + k]        = x + y;
                reg[j + k + ਕਦਮ] = x - y;
            }
        }
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < el_w; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

ਸਾਂਚਾ <ਪੂਰਨ N, ਪੂਰਨ NT, ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ launch_fwht_wide(ਅਚਲ T *       src,
                             ਦਸ਼ਮਲਵ *         dst,
                             ਅਚਲ int64_t   n_ਪੰਕਤੀਆਂ,
                             ਅਚਲ ਦਸ਼ਮਲਵ     scale,
                             dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ sycl::range<2> global(n_ਪੰਕਤੀਆਂ, NT);
    ਅਚਲ sycl::range<2> local(1, NT);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> smem(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_ਵਸਤੂ<2> ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             fwht_kernel_wide<N, NT, T>(src, dst, n_ਪੰਕਤੀਆਂ, scale, ਵਸਤੂ, get_ਸੂਚਕ(smem));
                         });
    });
}

ਸਾਂਚਾ <ਪੂਰਨ N, ਪੂਰਨ m, ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ kronecker_kernel(ਅਚਲ T * __restrict__ src,
                             ਦਸ਼ਮਲਵ * __restrict__ dst,
                             ਅਚਲ int64_t            n_ਪੰਕਤੀਆਂ,
                             ਅਚਲ ਦਸ਼ਮਲਵ              scale,
                             ਅਚਲ sycl::nd_ਵਸਤੂ<2> & ਵਸਤੂ) {
    static_assert(m == 12 || m == 20, "block size has to be 12 or 20.");

    ਅਚਲ sycl::sub_ਸਮੂਹ sg = ਵਸਤੂ.get_sub_ਸਮੂਹ();

    ਅਚਲ int64_t r = ਵਸਤੂ.get_global_id(0);
    ਜੇ (r >= n_ਪੰਕਤੀਆਂ) {
        ਵਾਪਸ;
    }

    src += r * N;
    dst += r * N;

    constexpr ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਸਮੂਹ = N / m;
    constexpr ਪੂਰਨ el_w             = ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਸਮੂਹ / WARP_ਮਾਪ;
    static_assert(el_w >= 1 && ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਸਮੂਹ % WARP_ਮਾਪ == 0, "blocks_per_group must be a multiple of WARP_SIZE");
    ਦਸ਼ਮਲਵ     reg[el_w * m];
    ਅਚਲ ਪੂਰਨ lane = sg.get_local_ਰੇਖਿਕ_id();

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < el_w; ++i) {
        ਅਚਲ ਪੂਰਨ b_idx = i * WARP_ਮਾਪ + lane;

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < m; ++j) {
            reg[i * m + j] = static_cast<ਦਸ਼ਮਲਵ>(src[b_idx * m + j]) * scale;
        }
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ b = 0; b < el_w; ++b) {
        ਦਸ਼ਮਲਵ z[m] = { 0.0f };

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i = 0; i < m; ++i) {
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ j = 0; j < m; ++j) {
                ਅਚਲ ਦਸ਼ਮਲਵ h = (m == 12 ? H12[j][i] : H20[j][i]);
                z[i] += reg[b * m + j] * h;
            }
        }

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i = 0; i < m; ++i) {
            reg[b * m + i] = z[i];
        }
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ h = 1; h < WARP_ਮਾਪ; h *= 2) {
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < el_w; ++j) {
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ k = 0; k < m; ++k) {
                ਅਚਲ ਦਸ਼ਮਲਵ val  = reg[j * m + k];
                ਅਚਲ ਦਸ਼ਮਲਵ val2 = dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(sg, val, h, WARP_ਮਾਪ);

                reg[j * m + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ h = WARP_ਮਾਪ; h < ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਸਮੂਹ; h *= 2) {
        ਅਚਲ ਪੂਰਨ ਕਦਮ = h / WARP_ਮਾਪ;
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < el_w; j += 2 * ਕਦਮ) {
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ s = 0; s < ਕਦਮ; ++s) {
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ k = 0; k < m; ++k) {
                    ਅਚਲ ਦਸ਼ਮਲਵ x = reg[(j + s) * m + k];
                    ਅਚਲ ਦਸ਼ਮਲਵ y = reg[(j + s + ਕਦਮ) * m + k];

                    reg[(j + s) * m + k]        = x + y;
                    reg[(j + s + ਕਦਮ) * m + k] = x - y;
                }
            }
        }
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < el_w; ++i) {
        ਅਚਲ ਪੂਰਨ b_idx = i * WARP_ਮਾਪ + lane;
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ k = 0; k < m; ++k) {
            dst[b_idx * m + k] = reg[i * m + k];
        }
    }
}

ਸਾਂਚਾ <ਪੂਰਨ N, ਪੂਰਨ m, ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ launch_kronecker(ਅਚਲ T *       src,
                             ਦਸ਼ਮਲਵ *         dst,
                             ਅਚਲ int64_t   n_ਪੰਕਤੀਆਂ,
                             ਅਚਲ ਦਸ਼ਮਲਵ     scale,
                             dpct::ਕਤਾਰ_ptr stream) {
    constexpr ਪੂਰਨ ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਖੰਡ = 4;

    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (n_ਪੰਕਤੀਆਂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਖੰਡ - 1) / ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਖੰਡ;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ਅਚਲ sycl::range<2> global(ਅੰਕ_ਖੰਡਾਂ * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਖੰਡ, WARP_ਮਾਪ);
    ਅਚਲ sycl::range<2> local(ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਖੰਡ, WARP_ਮਾਪ);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_ਵਸਤੂ<2> ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             kronecker_kernel<N, m, T>(src, dst, n_ਪੰਕਤੀਆਂ, scale, ਵਸਤੂ);
                         });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਤਾਰਕਿਕ ggml_sycl_op_fwht_impl(ggml_backend_sycl_context & ctx, ਅਚਲ ggml_tensor * src, ggml_tensor * dst) {
    ਅਚਲ ਪੂਰਨ     n    = (ਪੂਰਨ) src->ne[0];
    ਅਚਲ int64_t ਪੰਕਤੀਆਂ = ggml_nrows(src);

    ਅਚਲ T *       src_d  = (ਅਚਲ T *) src->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ *         dst_d  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;
    dpct::ਕਤਾਰ_ptr stream = ctx.stream();

    ਅਚਲ ਦਸ਼ਮਲਵ scale = 1.0f / std::ਵਰਗਮੂਲ((ਦਸ਼ਮਲਵ) n);

    ਚੋਣ (n) {
        ਹਾਲਤ 64:
            launch_fwht<64>(src_d, dst_d, ਪੰਕਤੀਆਂ, scale, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ 128:
            launch_fwht<128>(src_d, dst_d, ਪੰਕਤੀਆਂ, scale, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ 256:
            launch_fwht<256>(src_d, dst_d, ਪੰਕਤੀਆਂ, scale, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ 512:
            launch_fwht<512>(src_d, dst_d, ਪੰਕਤੀਆਂ, scale, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ 384:
            launch_kronecker<384, 12>(src_d, dst_d, ਪੰਕਤੀਆਂ, scale, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ 768:
            launch_kronecker<768, 12>(src_d, dst_d, ਪੰਕਤੀਆਂ, scale, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ 640:
            launch_kronecker<640, 20>(src_d, dst_d, ਪੰਕਤੀਆਂ, scale, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ 1280:
            launch_kronecker<1280, 20>(src_d, dst_d, ਪੰਕਤੀਆਂ, scale, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ 1024:
            launch_fwht_wide<1024, 256>(src_d, dst_d, ਪੰਕਤੀਆਂ, scale, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ 2048:
            launch_fwht_wide<2048, 256>(src_d, dst_d, ਪੰਕਤੀਆਂ, scale, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ 4096:
            launch_fwht_wide<4096, 256>(src_d, dst_d, ਪੰਕਤੀਆਂ, scale, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ 8192:
            launch_fwht_wide<8192, 256>(src_d, dst_d, ਪੰਕਤੀਆਂ, scale, stream);
            ਵਾਪਸ ਸੱਚ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }
}

ਤਾਰਕਿਕ ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, ਅਚਲ ggml_tensor * src, ggml_tensor * dst) {
    ਜੇ (dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!ggml_are_same_shape(src, dst)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਚੋਣ (src->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            ਵਾਪਸ ggml_sycl_op_fwht_impl<ਦਸ਼ਮਲਵ>(ctx, src, dst);
        ਹਾਲਤ GGML_TYPE_F16:
            ਵਾਪਸ ggml_sycl_op_fwht_impl<sycl::half>(ctx, src, dst);
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }
}
