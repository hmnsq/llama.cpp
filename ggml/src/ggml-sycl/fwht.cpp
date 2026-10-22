#ଅନ୍ତର୍ଭାଵ "fwht.hpp"

#ଅନ୍ତର୍ଭାଵ <cmath>
#define P 1.0f
#define N -1.0f

// constant Hadamard matrix via Paley I construction
ସ୍ଥିର constexpr ଦଶମିକ H12[12][12] = {
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

ସ୍ଥିର constexpr ଦଶମିକ H20[20][20] = {
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

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା N, typename T>
ସ୍ଥିର ଶୂନ୍ୟ fwht_kernel(ଅଚଳ T * __restrict__ src, ଦଶମିକ * __restrict__ dst, ଅଚଳ int64_t n_rows,
                        ଅଚଳ ଦଶମିକ scale, ଅଚଳ sycl::nd_item<2> & item) {
    ଅଚଳ sycl::sub_group sg = item.get_sub_group();

    ଅଚଳ int64_t r = item.get_global_id(0);
    ଯଦି (r >= n_rows) {
        ଫେରନ୍ତୁ;
    }

    src += r * N;
    dst += r * N;

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା el_w = N / WARP_SIZE;
    static_assert(el_w >= 1 && N % WARP_SIZE == 0, "row must be a whole number of sub-group widths");

    ଦଶମିକ     reg[el_w];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା lane = sg.get_local_linear_id();

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < el_w; ++i) {
        reg[i] = static_cast<ଦଶମିକ>(src[i * WARP_SIZE + lane]) * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < el_w; ++j) {
            ଅଚଳ ଦଶମିକ val  = reg[j];
            ଅଚଳ ଦଶମିକ val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା h = WARP_SIZE; h < N; h *= 2) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା step = h / WARP_SIZE;
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < step; ++k) {
                ଅଚଳ ଦଶମିକ x = reg[j + k];
                ଅଚଳ ଦଶମିକ y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < el_w; ++i) {
        dst[i * WARP_SIZE + lane] = reg[i];
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା N, typename T>
ସ୍ଥିର ଶୂନ୍ୟ launch_fwht(ଅଚଳ T * src, ଦଶମିକ * dst, ଅଚଳ int64_t n_rows, ଅଚଳ ଦଶମିକ scale,
                        dpct::queue_ptr stream) {
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା rows_per_block = 4;

    ଅଚଳ int64_t ଅଙ୍କ_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ଅଚଳ sycl::range<2> global(ଅଙ୍କ_blocks * rows_per_block, WARP_SIZE);
    ଅଚଳ sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel<N, T>(src, dst, n_rows, scale, item);
                         });
}

// Wide blocks: one row per work-group instead of per sub-group, so each work-item
// keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
// still shuffle; those up to NT go through work-group local memory; the rest stay
// in registers.
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା N, ପୂର୍ଣ୍ଣସଂଖ୍ୟା NT, typename T>
ସ୍ଥିର ଶୂନ୍ୟ fwht_kernel_wide(ଅଚଳ T * __restrict__ src,
                             ଦଶମିକ * __restrict__ dst,
                             ଅଚଳ int64_t            n_rows,
                             ଅଚଳ ଦଶମିକ              scale,
                             ଅଚଳ sycl::nd_item<2> & item,
                             ଦଶମିକ *                  smem) {
    ଅଚଳ int64_t r = item.get_global_id(0);
    ଯଦି (r >= n_rows) {
        ଫେରନ୍ତୁ;
    }

    src += r * N;
    dst += r * N;

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା el_w = N / NT;
    static_assert(el_w >= 1 && N % NT == 0, "row must be a whole number of work-group widths");

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tid = item.get_local_id(1);

    ଦଶମିକ reg[el_w];
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < el_w; ++i) {
        reg[i] = static_cast<ଦଶମିକ>(src[i * NT + tid]) * scale;
    }

    ଅଚଳ sycl::sub_group sg   = item.get_sub_group();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା             lane = sg.get_local_linear_id();

    // Butterflies inside the sub-group, same pattern as the narrow kernel.
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < el_w; ++j) {
            ଅଚଳ ଦଶମିକ val  = reg[j];
            ଅଚଳ ଦଶମିକ val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies from the sub-group width up to NT: the partner lane is outside
    // this sub-group, so it goes through work-group local memory instead of a shuffle.
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା h = WARP_SIZE; h < NT; h *= 2) {
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < el_w; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < el_w; ++j) {
            ଅଚଳ ଦଶମିକ val  = reg[j];
            ଅଚଳ ଦଶମିକ val2 = smem[j * NT + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    // Butterflies across registers: h is a multiple of NT, so the partner of element
    // i*NT + tid lives in reg[i + h/NT] on the same work-item.
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା h = NT; h < N; h *= 2) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା step = h / NT;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < el_w; j += 2 * step) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < step; ++k) {
                ଅଚଳ ଦଶମିକ x = reg[j + k];
                ଅଚଳ ଦଶମିକ y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < el_w; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା N, ପୂର୍ଣ୍ଣସଂଖ୍ୟା NT, typename T>
ସ୍ଥିର ଶୂନ୍ୟ launch_fwht_wide(ଅଚଳ T *       src,
                             ଦଶମିକ *         dst,
                             ଅଚଳ int64_t   n_rows,
                             ଅଚଳ ଦଶମିକ     scale,
                             dpct::queue_ptr stream) {
    ଅଚଳ sycl::range<2> global(n_rows, NT);
    ଅଚଳ sycl::range<2> local(1, NT);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<ଦଶମିକ, 1> smem(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel_wide<N, NT, T>(src, dst, n_rows, scale, item, get_pointer(smem));
                         });
    });
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା N, ପୂର୍ଣ୍ଣସଂଖ୍ୟା m, typename T>
ସ୍ଥିର ଶୂନ୍ୟ kronecker_kernel(ଅଚଳ T * __restrict__ src,
                             ଦଶମିକ * __restrict__ dst,
                             ଅଚଳ int64_t            n_rows,
                             ଅଚଳ ଦଶମିକ              scale,
                             ଅଚଳ sycl::nd_item<2> & item) {
    static_assert(m == 12 || m == 20, "block size has to be 12 or 20.");

    ଅଚଳ sycl::sub_group sg = item.get_sub_group();

    ଅଚଳ int64_t r = item.get_global_id(0);
    ଯଦି (r >= n_rows) {
        ଫେରନ୍ତୁ;
    }

    src += r * N;
    dst += r * N;

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_group = N / m;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା el_w             = blocks_per_group / WARP_SIZE;
    static_assert(el_w >= 1 && blocks_per_group % WARP_SIZE == 0, "blocks_per_group must be a multiple of WARP_SIZE");
    ଦଶମିକ     reg[el_w * m];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା lane = sg.get_local_linear_id();

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < el_w; ++i) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା b_idx = i * WARP_SIZE + lane;

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < m; ++j) {
            reg[i * m + j] = static_cast<ଦଶମିକ>(src[b_idx * m + j]) * scale;
        }
    }

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା b = 0; b < el_w; ++b) {
        ଦଶମିକ z[m] = { 0.0f };

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < m; ++i) {
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < m; ++j) {
                ଅଚଳ ଦଶମିକ h = (m == 12 ? H12[j][i] : H20[j][i]);
                z[i] += reg[b * m + j] * h;
            }
        }

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < m; ++i) {
            reg[b * m + i] = z[i];
        }
    }

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < el_w; ++j) {
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < m; ++k) {
                ଅଚଳ ଦଶମିକ val  = reg[j * m + k];
                ଅଚଳ ଦଶମିକ val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

                reg[j * m + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା h = WARP_SIZE; h < blocks_per_group; h *= 2) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା step = h / WARP_SIZE;
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < step; ++s) {
#pragma unroll
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < m; ++k) {
                    ଅଚଳ ଦଶମିକ x = reg[(j + s) * m + k];
                    ଅଚଳ ଦଶମିକ y = reg[(j + s + step) * m + k];

                    reg[(j + s) * m + k]        = x + y;
                    reg[(j + s + step) * m + k] = x - y;
                }
            }
        }
    }

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < el_w; ++i) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା b_idx = i * WARP_SIZE + lane;
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < m; ++k) {
            dst[b_idx * m + k] = reg[i * m + k];
        }
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା N, ପୂର୍ଣ୍ଣସଂଖ୍ୟା m, typename T>
ସ୍ଥିର ଶୂନ୍ୟ launch_kronecker(ଅଚଳ T *       src,
                             ଦଶମିକ *         dst,
                             ଅଚଳ int64_t   n_rows,
                             ଅଚଳ ଦଶମିକ     scale,
                             dpct::queue_ptr stream) {
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା rows_per_block = 4;

    ଅଚଳ int64_t ଅଙ୍କ_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ଅଚଳ sycl::range<2> global(ଅଙ୍କ_blocks * rows_per_block, WARP_SIZE);
    ଅଚଳ sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             kronecker_kernel<N, m, T>(src, dst, n_rows, scale, item);
                         });
}

template <typename T>
ସ୍ଥିର ତାର୍କିକ ggml_sycl_op_fwht_impl(ggml_backend_sycl_context & ctx, ଅଚଳ ggml_tensor * src, ggml_tensor * dst) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     n    = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) src->ne[0];
    ଅଚଳ int64_t rows = ggml_nrows(src);

    ଅଚଳ T *       src_d  = (ଅଚଳ T *) src->data;
    ଦଶମିକ *         dst_d  = (ଦଶମିକ *) dst->data;
    dpct::queue_ptr stream = ctx.stream();

    ଅଚଳ ଦଶମିକ scale = 1.0f / std::sqrt((ଦଶମିକ) n);

    ବାଛ (n) {
        ସ୍ଥିତି 64:
            launch_fwht<64>(src_d, dst_d, rows, scale, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି 128:
            launch_fwht<128>(src_d, dst_d, rows, scale, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି 256:
            launch_fwht<256>(src_d, dst_d, rows, scale, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି 512:
            launch_fwht<512>(src_d, dst_d, rows, scale, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି 384:
            launch_kronecker<384, 12>(src_d, dst_d, rows, scale, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି 768:
            launch_kronecker<768, 12>(src_d, dst_d, rows, scale, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି 640:
            launch_kronecker<640, 20>(src_d, dst_d, rows, scale, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି 1280:
            launch_kronecker<1280, 20>(src_d, dst_d, rows, scale, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି 1024:
            launch_fwht_wide<1024, 256>(src_d, dst_d, rows, scale, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି 2048:
            launch_fwht_wide<2048, 256>(src_d, dst_d, rows, scale, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି 4096:
            launch_fwht_wide<4096, 256>(src_d, dst_d, rows, scale, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି 8192:
            launch_fwht_wide<8192, 256>(src_d, dst_d, rows, scale, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
}

ତାର୍କିକ ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, ଅଚଳ ggml_tensor * src, ggml_tensor * dst) {
    ଯଦି (dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!ggml_are_same_shape(src, dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ବାଛ (src->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            ଫେରନ୍ତୁ ggml_sycl_op_fwht_impl<ଦଶମିକ>(ctx, src, dst);
        ସ୍ଥିତି GGML_TYPE_F16:
            ଫେରନ୍ତୁ ggml_sycl_op_fwht_impl<sycl::half>(ctx, src, dst);
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
}
