#அந்தர்பாவ "fwht.hpp"

#அந்தர்பாவ <cmath>
#define P 1.0f
#define N -1.0f

// constant Hadamard matrix via Paley I construction
நிலையான constexpr தசம H12[12][12] = {
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

நிலையான constexpr தசம H20[20][20] = {
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

template <முழு N, typename T>
நிலையான வெறுமை fwht_kernel(மாறிலி T * __restrict__ src, தசம * __restrict__ dst, மாறிலி int64_t n_rows,
                        மாறிலி தசம scale, மாறிலி sycl::nd_item<2> & item) {
    மாறிலி sycl::sub_group sg = item.get_sub_group();

    மாறிலி int64_t r = item.get_global_id(0);
    எனில் (r >= n_rows) {
        திரும்பு;
    }

    src += r * N;
    dst += r * N;

    constexpr முழு el_w = N / WARP_SIZE;
    static_assert(el_w >= 1 && N % WARP_SIZE == 0, "row must be a whole number of sub-group widths");

    தசம     reg[el_w];
    மாறிலி முழு lane = sg.get_local_linear_id();

#pragma unroll
    க்கு (முழு i = 0; i < el_w; ++i) {
        reg[i] = static_cast<தசம>(src[i * WARP_SIZE + lane]) * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#pragma unroll
    க்கு (முழு h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        க்கு (முழு j = 0; j < el_w; ++j) {
            மாறிலி தசம val  = reg[j];
            மாறிலி தசம val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#pragma unroll
    க்கு (முழு h = WARP_SIZE; h < N; h *= 2) {
        மாறிலி முழு step = h / WARP_SIZE;
#pragma unroll
        க்கு (முழு j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            க்கு (முழு k = 0; k < step; ++k) {
                மாறிலி தசம x = reg[j + k];
                மாறிலி தசம y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    க்கு (முழு i = 0; i < el_w; ++i) {
        dst[i * WARP_SIZE + lane] = reg[i];
    }
}

template <முழு N, typename T>
நிலையான வெறுமை launch_fwht(மாறிலி T * src, தசம * dst, மாறிலி int64_t n_rows, மாறிலி தசம scale,
                        dpct::queue_ptr stream) {
    constexpr முழு rows_per_block = 4;

    மாறிலி int64_t இலக்கம்_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    மாறிலி sycl::range<2> global(இலக்கம்_blocks * rows_per_block, WARP_SIZE);
    மாறிலி sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel<N, T>(src, dst, n_rows, scale, item);
                         });
}

// Wide blocks: one row per work-group instead of per sub-group, so each work-item
// keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
// still shuffle; those up to NT go through work-group local memory; the rest stay
// in registers.
template <முழு N, முழு NT, typename T>
நிலையான வெறுமை fwht_kernel_wide(மாறிலி T * __restrict__ src,
                             தசம * __restrict__ dst,
                             மாறிலி int64_t            n_rows,
                             மாறிலி தசம              scale,
                             மாறிலி sycl::nd_item<2> & item,
                             தசம *                  smem) {
    மாறிலி int64_t r = item.get_global_id(0);
    எனில் (r >= n_rows) {
        திரும்பு;
    }

    src += r * N;
    dst += r * N;

    constexpr முழு el_w = N / NT;
    static_assert(el_w >= 1 && N % NT == 0, "row must be a whole number of work-group widths");

    மாறிலி முழு tid = item.get_local_id(1);

    தசம reg[el_w];
#pragma unroll
    க்கு (முழு i = 0; i < el_w; ++i) {
        reg[i] = static_cast<தசம>(src[i * NT + tid]) * scale;
    }

    மாறிலி sycl::sub_group sg   = item.get_sub_group();
    மாறிலி முழு             lane = sg.get_local_linear_id();

    // Butterflies inside the sub-group, same pattern as the narrow kernel.
#pragma unroll
    க்கு (முழு h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        க்கு (முழு j = 0; j < el_w; ++j) {
            மாறிலி தசம val  = reg[j];
            மாறிலி தசம val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies from the sub-group width up to NT: the partner lane is outside
    // this sub-group, so it goes through work-group local memory instead of a shuffle.
    க்கு (முழு h = WARP_SIZE; h < NT; h *= 2) {
#pragma unroll
        க்கு (முழு j = 0; j < el_w; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        க்கு (முழு j = 0; j < el_w; ++j) {
            மாறிலி தசம val  = reg[j];
            மாறிலி தசம val2 = smem[j * NT + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    // Butterflies across registers: h is a multiple of NT, so the partner of element
    // i*NT + tid lives in reg[i + h/NT] on the same work-item.
    க்கு (முழு h = NT; h < N; h *= 2) {
        மாறிலி முழு step = h / NT;
        க்கு (முழு j = 0; j < el_w; j += 2 * step) {
            க்கு (முழு k = 0; k < step; ++k) {
                மாறிலி தசம x = reg[j + k];
                மாறிலி தசம y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    க்கு (முழு i = 0; i < el_w; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

template <முழு N, முழு NT, typename T>
நிலையான வெறுமை launch_fwht_wide(மாறிலி T *       src,
                             தசம *         dst,
                             மாறிலி int64_t   n_rows,
                             மாறிலி தசம     scale,
                             dpct::queue_ptr stream) {
    மாறிலி sycl::range<2> global(n_rows, NT);
    மாறிலி sycl::range<2> local(1, NT);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<தசம, 1> smem(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel_wide<N, NT, T>(src, dst, n_rows, scale, item, get_pointer(smem));
                         });
    });
}

template <முழு N, முழு m, typename T>
நிலையான வெறுமை kronecker_kernel(மாறிலி T * __restrict__ src,
                             தசம * __restrict__ dst,
                             மாறிலி int64_t            n_rows,
                             மாறிலி தசம              scale,
                             மாறிலி sycl::nd_item<2> & item) {
    static_assert(m == 12 || m == 20, "block size has to be 12 or 20.");

    மாறிலி sycl::sub_group sg = item.get_sub_group();

    மாறிலி int64_t r = item.get_global_id(0);
    எனில் (r >= n_rows) {
        திரும்பு;
    }

    src += r * N;
    dst += r * N;

    constexpr முழு blocks_per_group = N / m;
    constexpr முழு el_w             = blocks_per_group / WARP_SIZE;
    static_assert(el_w >= 1 && blocks_per_group % WARP_SIZE == 0, "blocks_per_group must be a multiple of WARP_SIZE");
    தசம     reg[el_w * m];
    மாறிலி முழு lane = sg.get_local_linear_id();

#pragma unroll
    க்கு (முழு i = 0; i < el_w; ++i) {
        மாறிலி முழு b_idx = i * WARP_SIZE + lane;

#pragma unroll
        க்கு (முழு j = 0; j < m; ++j) {
            reg[i * m + j] = static_cast<தசம>(src[b_idx * m + j]) * scale;
        }
    }

#pragma unroll
    க்கு (முழு b = 0; b < el_w; ++b) {
        தசம z[m] = { 0.0f };

#pragma unroll
        க்கு (முழு i = 0; i < m; ++i) {
#pragma unroll
            க்கு (முழு j = 0; j < m; ++j) {
                மாறிலி தசம h = (m == 12 ? H12[j][i] : H20[j][i]);
                z[i] += reg[b * m + j] * h;
            }
        }

#pragma unroll
        க்கு (முழு i = 0; i < m; ++i) {
            reg[b * m + i] = z[i];
        }
    }

#pragma unroll
    க்கு (முழு h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        க்கு (முழு j = 0; j < el_w; ++j) {
#pragma unroll
            க்கு (முழு k = 0; k < m; ++k) {
                மாறிலி தசம val  = reg[j * m + k];
                மாறிலி தசம val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

                reg[j * m + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#pragma unroll
    க்கு (முழு h = WARP_SIZE; h < blocks_per_group; h *= 2) {
        மாறிலி முழு step = h / WARP_SIZE;
#pragma unroll
        க்கு (முழு j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            க்கு (முழு s = 0; s < step; ++s) {
#pragma unroll
                க்கு (முழு k = 0; k < m; ++k) {
                    மாறிலி தசம x = reg[(j + s) * m + k];
                    மாறிலி தசம y = reg[(j + s + step) * m + k];

                    reg[(j + s) * m + k]        = x + y;
                    reg[(j + s + step) * m + k] = x - y;
                }
            }
        }
    }

#pragma unroll
    க்கு (முழு i = 0; i < el_w; ++i) {
        மாறிலி முழு b_idx = i * WARP_SIZE + lane;
#pragma unroll
        க்கு (முழு k = 0; k < m; ++k) {
            dst[b_idx * m + k] = reg[i * m + k];
        }
    }
}

template <முழு N, முழு m, typename T>
நிலையான வெறுமை launch_kronecker(மாறிலி T *       src,
                             தசம *         dst,
                             மாறிலி int64_t   n_rows,
                             மாறிலி தசம     scale,
                             dpct::queue_ptr stream) {
    constexpr முழு rows_per_block = 4;

    மாறிலி int64_t இலக்கம்_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    மாறிலி sycl::range<2> global(இலக்கம்_blocks * rows_per_block, WARP_SIZE);
    மாறிலி sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             kronecker_kernel<N, m, T>(src, dst, n_rows, scale, item);
                         });
}

template <typename T>
நிலையான தருக்க ggml_sycl_op_fwht_impl(ggml_backend_sycl_context & ctx, மாறிலி ggml_tensor * src, ggml_tensor * dst) {
    மாறிலி முழு     n    = (முழு) src->ne[0];
    மாறிலி int64_t rows = ggml_nrows(src);

    மாறிலி T *       src_d  = (மாறிலி T *) src->data;
    தசம *         dst_d  = (தசம *) dst->data;
    dpct::queue_ptr stream = ctx.stream();

    மாறிலி தசம scale = 1.0f / std::sqrt((தசம) n);

    தேர்வு (n) {
        நிலை 64:
            launch_fwht<64>(src_d, dst_d, rows, scale, stream);
            திரும்பு உண்மை;
        நிலை 128:
            launch_fwht<128>(src_d, dst_d, rows, scale, stream);
            திரும்பு உண்மை;
        நிலை 256:
            launch_fwht<256>(src_d, dst_d, rows, scale, stream);
            திரும்பு உண்மை;
        நிலை 512:
            launch_fwht<512>(src_d, dst_d, rows, scale, stream);
            திரும்பு உண்மை;
        நிலை 384:
            launch_kronecker<384, 12>(src_d, dst_d, rows, scale, stream);
            திரும்பு உண்மை;
        நிலை 768:
            launch_kronecker<768, 12>(src_d, dst_d, rows, scale, stream);
            திரும்பு உண்மை;
        நிலை 640:
            launch_kronecker<640, 20>(src_d, dst_d, rows, scale, stream);
            திரும்பு உண்மை;
        நிலை 1280:
            launch_kronecker<1280, 20>(src_d, dst_d, rows, scale, stream);
            திரும்பு உண்மை;
        நிலை 1024:
            launch_fwht_wide<1024, 256>(src_d, dst_d, rows, scale, stream);
            திரும்பு உண்மை;
        நிலை 2048:
            launch_fwht_wide<2048, 256>(src_d, dst_d, rows, scale, stream);
            திரும்பு உண்மை;
        நிலை 4096:
            launch_fwht_wide<4096, 256>(src_d, dst_d, rows, scale, stream);
            திரும்பு உண்மை;
        நிலை 8192:
            launch_fwht_wide<8192, 256>(src_d, dst_d, rows, scale, stream);
            திரும்பு உண்மை;
        இயல்பு:
            திரும்பு பொய்;
    }
}

தருக்க ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, மாறிலி ggml_tensor * src, ggml_tensor * dst) {
    எனில் (dst->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }
    எனில் (!ggml_are_same_shape(src, dst)) {
        திரும்பு பொய்;
    }
    எனில் (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        திரும்பு பொய்;
    }

    தேர்வு (src->type) {
        நிலை GGML_TYPE_F32:
            திரும்பு ggml_sycl_op_fwht_impl<தசம>(ctx, src, dst);
        நிலை GGML_TYPE_F16:
            திரும்பு ggml_sycl_op_fwht_impl<sycl::half>(ctx, src, dst);
        இயல்பு:
            திரும்பு பொய்;
    }
}
