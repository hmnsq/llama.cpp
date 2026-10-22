#ಅನ್ತರ್ಭಾವ "fwht.hpp"

#ಅನ್ತರ್ಭಾವ <cmath>
#define P 1.0f
#define N -1.0f

// constant Hadamard matrix via Paley I construction
ಸ್ಥಿರ constexpr ಭಿನ್ನ H12[12][12] = {
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

ಸ್ಥಿರ constexpr ಭಿನ್ನ H20[20][20] = {
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

#undef P
#undef N

template <ಪೂರ್ಣಾಙ್ಕ N, typename T>
ಸ್ಥಿರ ಶೂನ್ಯ fwht_kernel(ಧ್ರುವ T * __restrict__ src, ಭಿನ್ನ * __restrict__ dst, ಧ್ರುವ int64_t n_rows,
                        ಧ್ರುವ ಭಿನ್ನ scale, ಧ್ರುವ sycl::nd_item<2> & item) {
    ಧ್ರುವ sycl::sub_group sg = item.get_sub_group();

    ಧ್ರುವ int64_t r = item.get_global_id(0);
    ಯದಿ (r >= n_rows) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    src += r * N;
    dst += r * N;

    constexpr ಪೂರ್ಣಾಙ್ಕ el_w = N / WARP_SIZE;
    static_assert(el_w >= 1 && N % WARP_SIZE == 0, "row must be a whole number of sub-group widths");

    ಭಿನ್ನ     reg[el_w];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ lane = sg.get_local_linear_id();

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < el_w; ++i) {
        reg[i] = static_cast<ಭಿನ್ನ>(src[i * WARP_SIZE + lane]) * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < el_w; ++j) {
            ಧ್ರುವ ಭಿನ್ನ val  = reg[j];
            ಧ್ರುವ ಭಿನ್ನ val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ h = WARP_SIZE; h < N; h *= 2) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ step = h / WARP_SIZE;
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = 0; k < step; ++k) {
                ಧ್ರುವ ಭಿನ್ನ x = reg[j + k];
                ಧ್ರುವ ಭಿನ್ನ y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < el_w; ++i) {
        dst[i * WARP_SIZE + lane] = reg[i];
    }
}

template <ಪೂರ್ಣಾಙ್ಕ N, typename T>
ಸ್ಥಿರ ಶೂನ್ಯ launch_fwht(ಧ್ರುವ T * src, ಭಿನ್ನ * dst, ಧ್ರುವ int64_t n_rows, ಧ್ರುವ ಭಿನ್ನ scale,
                        dpct::queue_ptr stream) {
    constexpr ಪೂರ್ಣಾಙ್ಕ rows_per_block = 4;

    ಧ್ರುವ int64_t ಅಂಕೆ_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ಧ್ರುವ sycl::range<2> global(ಅಂಕೆ_blocks * rows_per_block, WARP_SIZE);
    ಧ್ರುವ sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel<N, T>(src, dst, n_rows, scale, item);
                         });
}

// Wide blocks: one row per work-group instead of per sub-group, so each work-item
// keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
// still shuffle; those up to NT go through work-group local memory; the rest stay
// in registers.
template <ಪೂರ್ಣಾಙ್ಕ N, ಪೂರ್ಣಾಙ್ಕ NT, typename T>
ಸ್ಥಿರ ಶೂನ್ಯ fwht_kernel_wide(ಧ್ರುವ T * __restrict__ src,
                             ಭಿನ್ನ * __restrict__ dst,
                             ಧ್ರುವ int64_t            n_rows,
                             ಧ್ರುವ ಭಿನ್ನ              scale,
                             ಧ್ರುವ sycl::nd_item<2> & item,
                             ಭಿನ್ನ *                  smem) {
    ಧ್ರುವ int64_t r = item.get_global_id(0);
    ಯದಿ (r >= n_rows) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    src += r * N;
    dst += r * N;

    constexpr ಪೂರ್ಣಾಙ್ಕ el_w = N / NT;
    static_assert(el_w >= 1 && N % NT == 0, "row must be a whole number of work-group widths");

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ tid = item.get_local_id(1);

    ಭಿನ್ನ reg[el_w];
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < el_w; ++i) {
        reg[i] = static_cast<ಭಿನ್ನ>(src[i * NT + tid]) * scale;
    }

    ಧ್ರುವ sycl::sub_group sg   = item.get_sub_group();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ             lane = sg.get_local_linear_id();

    // Butterflies inside the sub-group, same pattern as the narrow kernel.
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < el_w; ++j) {
            ಧ್ರುವ ಭಿನ್ನ val  = reg[j];
            ಧ್ರುವ ಭಿನ್ನ val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies from the sub-group width up to NT: the partner lane is outside
    // this sub-group, so it goes through work-group local memory instead of a shuffle.
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ h = WARP_SIZE; h < NT; h *= 2) {
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < el_w; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < el_w; ++j) {
            ಧ್ರುವ ಭಿನ್ನ val  = reg[j];
            ಧ್ರುವ ಭಿನ್ನ val2 = smem[j * NT + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    // Butterflies across registers: h is a multiple of NT, so the partner of element
    // i*NT + tid lives in reg[i + h/NT] on the same work-item.
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ h = NT; h < N; h *= 2) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ step = h / NT;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < el_w; j += 2 * step) {
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = 0; k < step; ++k) {
                ಧ್ರುವ ಭಿನ್ನ x = reg[j + k];
                ಧ್ರುವ ಭಿನ್ನ y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < el_w; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

template <ಪೂರ್ಣಾಙ್ಕ N, ಪೂರ್ಣಾಙ್ಕ NT, typename T>
ಸ್ಥಿರ ಶೂನ್ಯ launch_fwht_wide(ಧ್ರುವ T *       src,
                             ಭಿನ್ನ *         dst,
                             ಧ್ರುವ int64_t   n_rows,
                             ಧ್ರುವ ಭಿನ್ನ     scale,
                             dpct::queue_ptr stream) {
    ಧ್ರುವ sycl::range<2> global(n_rows, NT);
    ಧ್ರುವ sycl::range<2> local(1, NT);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<ಭಿನ್ನ, 1> smem(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel_wide<N, NT, T>(src, dst, n_rows, scale, item, get_pointer(smem));
                         });
    });
}

template <ಪೂರ್ಣಾಙ್ಕ N, ಪೂರ್ಣಾಙ್ಕ m, typename T>
ಸ್ಥಿರ ಶೂನ್ಯ kronecker_kernel(ಧ್ರುವ T * __restrict__ src,
                             ಭಿನ್ನ * __restrict__ dst,
                             ಧ್ರುವ int64_t            n_rows,
                             ಧ್ರುವ ಭಿನ್ನ              scale,
                             ಧ್ರುವ sycl::nd_item<2> & item) {
    static_assert(m == 12 || m == 20, "block size has to be 12 or 20.");

    ಧ್ರುವ sycl::sub_group sg = item.get_sub_group();

    ಧ್ರುವ int64_t r = item.get_global_id(0);
    ಯದಿ (r >= n_rows) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    src += r * N;
    dst += r * N;

    constexpr ಪೂರ್ಣಾಙ್ಕ blocks_per_group = N / m;
    constexpr ಪೂರ್ಣಾಙ್ಕ el_w             = blocks_per_group / WARP_SIZE;
    static_assert(el_w >= 1 && blocks_per_group % WARP_SIZE == 0, "blocks_per_group must be a multiple of WARP_SIZE");
    ಭಿನ್ನ     reg[el_w * m];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ lane = sg.get_local_linear_id();

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < el_w; ++i) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ b_idx = i * WARP_SIZE + lane;

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < m; ++j) {
            reg[i * m + j] = static_cast<ಭಿನ್ನ>(src[b_idx * m + j]) * scale;
        }
    }

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ b = 0; b < el_w; ++b) {
        ಭಿನ್ನ z[m] = { 0.0f };

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < m; ++i) {
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < m; ++j) {
                ಧ್ರುವ ಭಿನ್ನ h = (m == 12 ? H12[j][i] : H20[j][i]);
                z[i] += reg[b * m + j] * h;
            }
        }

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < m; ++i) {
            reg[b * m + i] = z[i];
        }
    }

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < el_w; ++j) {
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = 0; k < m; ++k) {
                ಧ್ರುವ ಭಿನ್ನ val  = reg[j * m + k];
                ಧ್ರುವ ಭಿನ್ನ val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

                reg[j * m + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ h = WARP_SIZE; h < blocks_per_group; h *= 2) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ step = h / WARP_SIZE;
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ s = 0; s < step; ++s) {
#pragma unroll
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = 0; k < m; ++k) {
                    ಧ್ರುವ ಭಿನ್ನ x = reg[(j + s) * m + k];
                    ಧ್ರುವ ಭಿನ್ನ y = reg[(j + s + step) * m + k];

                    reg[(j + s) * m + k]        = x + y;
                    reg[(j + s + step) * m + k] = x - y;
                }
            }
        }
    }

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < el_w; ++i) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ b_idx = i * WARP_SIZE + lane;
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = 0; k < m; ++k) {
            dst[b_idx * m + k] = reg[i * m + k];
        }
    }
}

template <ಪೂರ್ಣಾಙ್ಕ N, ಪೂರ್ಣಾಙ್ಕ m, typename T>
ಸ್ಥಿರ ಶೂನ್ಯ launch_kronecker(ಧ್ರುವ T *       src,
                             ಭಿನ್ನ *         dst,
                             ಧ್ರುವ int64_t   n_rows,
                             ಧ್ರುವ ಭಿನ್ನ     scale,
                             dpct::queue_ptr stream) {
    constexpr ಪೂರ್ಣಾಙ್ಕ rows_per_block = 4;

    ಧ್ರುವ int64_t ಅಂಕೆ_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ಧ್ರುವ sycl::range<2> global(ಅಂಕೆ_blocks * rows_per_block, WARP_SIZE);
    ಧ್ರುವ sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             kronecker_kernel<N, m, T>(src, dst, n_rows, scale, item);
                         });
}

template <typename T>
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_op_fwht_impl(ggml_backend_sycl_context & ctx, ಧ್ರುವ ggml_tensor * src, ggml_tensor * dst) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     n    = (ಪೂರ್ಣಾಙ್ಕ) src->ne[0];
    ಧ್ರುವ int64_t rows = ggml_nrows(src);

    ಧ್ರುವ T *       src_d  = (ಧ್ರುವ T *) src->data;
    ಭಿನ್ನ *         dst_d  = (ಭಿನ್ನ *) dst->data;
    dpct::queue_ptr stream = ctx.stream();

    ಧ್ರುವ ಭಿನ್ನ scale = 1.0f / std::sqrt((ಭಿನ್ನ) n);

    ವಿಕಲ್ಪ (n) {
        ಪಕ್ಷ 64:
            launch_fwht<64>(src_d, dst_d, rows, scale, stream);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಪಕ್ಷ 128:
            launch_fwht<128>(src_d, dst_d, rows, scale, stream);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಪಕ್ಷ 256:
            launch_fwht<256>(src_d, dst_d, rows, scale, stream);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಪಕ್ಷ 512:
            launch_fwht<512>(src_d, dst_d, rows, scale, stream);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಪಕ್ಷ 384:
            launch_kronecker<384, 12>(src_d, dst_d, rows, scale, stream);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಪಕ್ಷ 768:
            launch_kronecker<768, 12>(src_d, dst_d, rows, scale, stream);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಪಕ್ಷ 640:
            launch_kronecker<640, 20>(src_d, dst_d, rows, scale, stream);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಪಕ್ಷ 1280:
            launch_kronecker<1280, 20>(src_d, dst_d, rows, scale, stream);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಪಕ್ಷ 1024:
            launch_fwht_wide<1024, 256>(src_d, dst_d, rows, scale, stream);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಪಕ್ಷ 2048:
            launch_fwht_wide<2048, 256>(src_d, dst_d, rows, scale, stream);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಪಕ್ಷ 4096:
            launch_fwht_wide<4096, 256>(src_d, dst_d, rows, scale, stream);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಪಕ್ಷ 8192:
            launch_fwht_wide<8192, 256>(src_d, dst_d, rows, scale, stream);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
}

ತಾರ್ಕಿಕ ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, ಧ್ರುವ ggml_tensor * src, ggml_tensor * dst) {
    ಯದಿ (dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!ggml_are_same_shape(src, dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ವಿಕಲ್ಪ (src->type) {
        ಪಕ್ಷ GGML_TYPE_F32:
            ಪ್ರತ್ಯಾಗಮ ggml_sycl_op_fwht_impl<ಭಿನ್ನ>(ctx, src, dst);
        ಪಕ್ಷ GGML_TYPE_F16:
            ಪ್ರತ್ಯಾಗಮ ggml_sycl_op_fwht_impl<sycl::half>(ctx, src, dst);
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
}
