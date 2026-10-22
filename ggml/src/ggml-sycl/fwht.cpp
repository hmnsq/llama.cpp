#ಅನ್ತರ್ಭಾವ "fwht.hpp"

#ಅನ್ತರ್ಭಾವ <cmath>
#ವ್ಯಾಖ್ಯಾನಿಸು P 1.0f
#ವ್ಯಾಖ್ಯಾನಿಸು N -1.0f

// constant Hadamard matrix via Paley I construction
ಸ್ಥಿರ constexpr ದಶಮಾಂಶ H12[12][12] = {
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

ಸ್ಥಿರ constexpr ದಶಮಾಂಶ H20[20][20] = {
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

#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು P
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು N

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ N, ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ fwht_kernel(ಸ್ಥಿರಾಂಕ T * __restrict__ src, ದಶಮಾಂಶ * __restrict__ dst, ಸ್ಥಿರಾಂಕ int64_t n_ಅಡ್ಡಸಾಲುಗಳು,
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<2> & ಅಂಶ) {
    ಸ್ಥಿರಾಂಕ sycl::sub_ಗುಂಪು sg = ಅಂಶ.get_sub_ಗುಂಪು();

    ಸ್ಥಿರಾಂಕ int64_t r = ಅಂಶ.get_global_id(0);
    ಆದರೆ (r >= n_ಅಡ್ಡಸಾಲುಗಳು) {
        ಹಿಂತಿರುಗಿ;
    }

    src += r * N;
    dst += r * N;

    constexpr ಪೂರ್ಣಾಂಕ el_w = N / WARP_ಗಾತ್ರ;
    static_assert(el_w >= 1 && N % WARP_ಗಾತ್ರ == 0, "row must be a whole number of sub-group widths");

    ದಶಮಾಂಶ     reg[el_w];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lane = sg.get_local_ರೇಖೀಯ_id();

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < el_w; ++i) {
        reg[i] = static_cast<ದಶಮಾಂಶ>(src[i * WARP_ಗಾತ್ರ + lane]) * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ h = 1; h < WARP_ಗಾತ್ರ; h *= 2) {
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < el_w; ++j) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val  = reg[j];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val2 = dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(sg, val, h, WARP_ಗಾತ್ರ);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ h = WARP_ಗಾತ್ರ; h < N; h *= 2) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = h / WARP_ಗಾತ್ರ;
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < el_w; j += 2 * ಹೆಜ್ಜೆ) {
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < ಹೆಜ್ಜೆ; ++k) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = reg[j + k];
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ y = reg[j + k + ಹೆಜ್ಜೆ];

                reg[j + k]        = x + y;
                reg[j + k + ಹೆಜ್ಜೆ] = x - y;
            }
        }
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < el_w; ++i) {
        dst[i * WARP_ಗಾತ್ರ + lane] = reg[i];
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ N, ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ launch_fwht(ಸ್ಥಿರಾಂಕ T * src, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ int64_t n_ಅಡ್ಡಸಾಲುಗಳು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale,
                        dpct::ಸರತಿ_ptr stream) {
    constexpr ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಖಂಡ = 4;

    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (n_ಅಡ್ಡಸಾಲುಗಳು + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಖಂಡ - 1) / ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಖಂಡ;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ಸ್ಥಿರಾಂಕ sycl::range<2> global(ಅಂಕೆ_ಖಂಡಗಳು * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಖಂಡ, WARP_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ sycl::range<2> local(ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಖಂಡ, WARP_ಗಾತ್ರ);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_ಅಂಶ<2> ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             fwht_kernel<N, T>(src, dst, n_ಅಡ್ಡಸಾಲುಗಳು, scale, ಅಂಶ);
                         });
}

// Wide blocks: one row per work-group instead of per sub-group, so each work-item
// keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
// still shuffle; those up to NT go through work-group local memory; the rest stay
// in registers.
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ N, ಪೂರ್ಣಾಂಕ NT, ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ fwht_kernel_wide(ಸ್ಥಿರಾಂಕ T * __restrict__ src,
                             ದಶಮಾಂಶ * __restrict__ dst,
                             ಸ್ಥಿರಾಂಕ int64_t            n_ಅಡ್ಡಸಾಲುಗಳು,
                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ              scale,
                             ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<2> & ಅಂಶ,
                             ದಶಮಾಂಶ *                  smem) {
    ಸ್ಥಿರಾಂಕ int64_t r = ಅಂಶ.get_global_id(0);
    ಆದರೆ (r >= n_ಅಡ್ಡಸಾಲುಗಳು) {
        ಹಿಂತಿರುಗಿ;
    }

    src += r * N;
    dst += r * N;

    constexpr ಪೂರ್ಣಾಂಕ el_w = N / NT;
    static_assert(el_w >= 1 && N % NT == 0, "row must be a whole number of work-group widths");

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ.get_local_id(1);

    ದಶಮಾಂಶ reg[el_w];
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < el_w; ++i) {
        reg[i] = static_cast<ದಶಮಾಂಶ>(src[i * NT + tid]) * scale;
    }

    ಸ್ಥಿರಾಂಕ sycl::sub_ಗುಂಪು sg   = ಅಂಶ.get_sub_ಗುಂಪು();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ             lane = sg.get_local_ರೇಖೀಯ_id();

    // Butterflies inside the sub-group, same pattern as the narrow kernel.
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ h = 1; h < WARP_ಗಾತ್ರ; h *= 2) {
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < el_w; ++j) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val  = reg[j];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val2 = dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(sg, val, h, WARP_ಗಾತ್ರ);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies from the sub-group width up to NT: the partner lane is outside
    // this sub-group, so it goes through work-group local memory instead of a shuffle.
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ h = WARP_ಗಾತ್ರ; h < NT; h *= 2) {
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < el_w; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        ಅಂಶ.barrier(sycl::access::fence_space::local_space);
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < el_w; ++j) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val  = reg[j];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val2 = smem[j * NT + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        ಅಂಶ.barrier(sycl::access::fence_space::local_space);
    }

    // Butterflies across registers: h is a multiple of NT, so the partner of element
    // i*NT + tid lives in reg[i + h/NT] on the same work-item.
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ h = NT; h < N; h *= 2) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = h / NT;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < el_w; j += 2 * ಹೆಜ್ಜೆ) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < ಹೆಜ್ಜೆ; ++k) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = reg[j + k];
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ y = reg[j + k + ಹೆಜ್ಜೆ];

                reg[j + k]        = x + y;
                reg[j + k + ಹೆಜ್ಜೆ] = x - y;
            }
        }
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < el_w; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ N, ಪೂರ್ಣಾಂಕ NT, ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ launch_fwht_wide(ಸ್ಥಿರಾಂಕ T *       src,
                             ದಶಮಾಂಶ *         dst,
                             ಸ್ಥಿರಾಂಕ int64_t   n_ಅಡ್ಡಸಾಲುಗಳು,
                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     scale,
                             dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ sycl::range<2> global(n_ಅಡ್ಡಸಾಲುಗಳು, NT);
    ಸ್ಥಿರಾಂಕ sycl::range<2> local(1, NT);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<ದಶಮಾಂಶ, 1> smem(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_ಅಂಶ<2> ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             fwht_kernel_wide<N, NT, T>(src, dst, n_ಅಡ್ಡಸಾಲುಗಳು, scale, ಅಂಶ, get_ಸೂಚಕ(smem));
                         });
    });
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ N, ಪೂರ್ಣಾಂಕ m, ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ kronecker_kernel(ಸ್ಥಿರಾಂಕ T * __restrict__ src,
                             ದಶಮಾಂಶ * __restrict__ dst,
                             ಸ್ಥಿರಾಂಕ int64_t            n_ಅಡ್ಡಸಾಲುಗಳು,
                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ              scale,
                             ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<2> & ಅಂಶ) {
    static_assert(m == 12 || m == 20, "block size has to be 12 or 20.");

    ಸ್ಥಿರಾಂಕ sycl::sub_ಗುಂಪು sg = ಅಂಶ.get_sub_ಗುಂಪು();

    ಸ್ಥಿರಾಂಕ int64_t r = ಅಂಶ.get_global_id(0);
    ಆದರೆ (r >= n_ಅಡ್ಡಸಾಲುಗಳು) {
        ಹಿಂತಿರುಗಿ;
    }

    src += r * N;
    dst += r * N;

    constexpr ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_ಗುಂಪು = N / m;
    constexpr ಪೂರ್ಣಾಂಕ el_w             = ಖಂಡಗಳು_ಪ್ರತಿ_ಗುಂಪು / WARP_ಗಾತ್ರ;
    static_assert(el_w >= 1 && ಖಂಡಗಳು_ಪ್ರತಿ_ಗುಂಪು % WARP_ಗಾತ್ರ == 0, "blocks_per_group must be a multiple of WARP_SIZE");
    ದಶಮಾಂಶ     reg[el_w * m];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lane = sg.get_local_ರೇಖೀಯ_id();

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < el_w; ++i) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ b_idx = i * WARP_ಗಾತ್ರ + lane;

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < m; ++j) {
            reg[i * m + j] = static_cast<ದಶಮಾಂಶ>(src[b_idx * m + j]) * scale;
        }
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < el_w; ++b) {
        ದಶಮಾಂಶ z[m] = { 0.0f };

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < m; ++i) {
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < m; ++j) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ h = (m == 12 ? H12[j][i] : H20[j][i]);
                z[i] += reg[b * m + j] * h;
            }
        }

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < m; ++i) {
            reg[b * m + i] = z[i];
        }
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ h = 1; h < WARP_ಗಾತ್ರ; h *= 2) {
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < el_w; ++j) {
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < m; ++k) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val  = reg[j * m + k];
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val2 = dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(sg, val, h, WARP_ಗಾತ್ರ);

                reg[j * m + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ h = WARP_ಗಾತ್ರ; h < ಖಂಡಗಳು_ಪ್ರತಿ_ಗುಂಪು; h *= 2) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = h / WARP_ಗಾತ್ರ;
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < el_w; j += 2 * ಹೆಜ್ಜೆ) {
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < ಹೆಜ್ಜೆ; ++s) {
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < m; ++k) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = reg[(j + s) * m + k];
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ y = reg[(j + s + ಹೆಜ್ಜೆ) * m + k];

                    reg[(j + s) * m + k]        = x + y;
                    reg[(j + s + ಹೆಜ್ಜೆ) * m + k] = x - y;
                }
            }
        }
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < el_w; ++i) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ b_idx = i * WARP_ಗಾತ್ರ + lane;
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < m; ++k) {
            dst[b_idx * m + k] = reg[i * m + k];
        }
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ N, ಪೂರ್ಣಾಂಕ m, ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ launch_kronecker(ಸ್ಥಿರಾಂಕ T *       src,
                             ದಶಮಾಂಶ *         dst,
                             ಸ್ಥಿರಾಂಕ int64_t   n_ಅಡ್ಡಸಾಲುಗಳು,
                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     scale,
                             dpct::ಸರತಿ_ptr stream) {
    constexpr ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಖಂಡ = 4;

    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (n_ಅಡ್ಡಸಾಲುಗಳು + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಖಂಡ - 1) / ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಖಂಡ;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ಸ್ಥಿರಾಂಕ sycl::range<2> global(ಅಂಕೆ_ಖಂಡಗಳು * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಖಂಡ, WARP_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ sycl::range<2> local(ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಖಂಡ, WARP_ಗಾತ್ರ);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_ಅಂಶ<2> ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             kronecker_kernel<N, m, T>(src, dst, n_ಅಡ್ಡಸಾಲುಗಳು, scale, ಅಂಶ);
                         });
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_op_fwht_impl(ggml_backend_sycl_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     n    = (ಪೂರ್ಣಾಂಕ) src->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲುಗಳು = ggml_nrows(src);

    ಸ್ಥಿರಾಂಕ T *       src_d  = (ಸ್ಥಿರಾಂಕ T *) src->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ *         dst_d  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;
    dpct::ಸರತಿ_ptr stream = ctx.stream();

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = 1.0f / std::ವರ್ಗಮೂಲ((ದಶಮಾಂಶ) n);

    ಆಯ್ಕೆ (n) {
        ಸ್ಥಿತಿ 64:
            launch_fwht<64>(src_d, dst_d, ಅಡ್ಡಸಾಲುಗಳು, scale, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ 128:
            launch_fwht<128>(src_d, dst_d, ಅಡ್ಡಸಾಲುಗಳು, scale, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ 256:
            launch_fwht<256>(src_d, dst_d, ಅಡ್ಡಸಾಲುಗಳು, scale, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ 512:
            launch_fwht<512>(src_d, dst_d, ಅಡ್ಡಸಾಲುಗಳು, scale, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ 384:
            launch_kronecker<384, 12>(src_d, dst_d, ಅಡ್ಡಸಾಲುಗಳು, scale, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ 768:
            launch_kronecker<768, 12>(src_d, dst_d, ಅಡ್ಡಸಾಲುಗಳು, scale, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ 640:
            launch_kronecker<640, 20>(src_d, dst_d, ಅಡ್ಡಸಾಲುಗಳು, scale, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ 1280:
            launch_kronecker<1280, 20>(src_d, dst_d, ಅಡ್ಡಸಾಲುಗಳು, scale, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ 1024:
            launch_fwht_wide<1024, 256>(src_d, dst_d, ಅಡ್ಡಸಾಲುಗಳು, scale, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ 2048:
            launch_fwht_wide<2048, 256>(src_d, dst_d, ಅಡ್ಡಸಾಲುಗಳು, scale, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ 4096:
            launch_fwht_wide<4096, 256>(src_d, dst_d, ಅಡ್ಡಸಾಲುಗಳು, scale, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ 8192:
            launch_fwht_wide<8192, 256>(src_d, dst_d, ಅಡ್ಡಸಾಲುಗಳು, scale, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

ತಾರ್ಕಿಕ ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src, ggml_tensor * dst) {
    ಆದರೆ (dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!ggml_are_same_shape(src, dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆಯ್ಕೆ (src->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            ಹಿಂತಿರುಗಿ ggml_sycl_op_fwht_impl<ದಶಮಾಂಶ>(ctx, src, dst);
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            ಹಿಂತಿರುಗಿ ggml_sycl_op_fwht_impl<sycl::half>(ctx, src, dst);
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}
