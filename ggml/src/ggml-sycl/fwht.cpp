#ضمن "fwht.hpp"

#ضمن <cmath>
#define P 1.0f
#define N -1.0f

// constant Hadamard matrix via Paley I construction
ساكن constexpr عائم H12[12][12] = {
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

ساكن constexpr عائم H20[20][20] = {
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

template <صحيح N, typename T>
ساكن فراغ fwht_kernel(ثابت T * __restrict__ src, عائم * __restrict__ dst, ثابت int64_t n_rows,
                        ثابت عائم scale, ثابت sycl::nd_item<2> & item) {
    ثابت sycl::sub_group sg = item.get_sub_group();

    ثابت int64_t r = item.get_global_id(0);
    ܐܢ (r >= n_rows) {
        ܦܢܐ;
    }

    src += r * N;
    dst += r * N;

    constexpr صحيح el_w = N / WARP_SIZE;
    static_assert(el_w >= 1 && N % WARP_SIZE == 0, "row must be a whole number of sub-group widths");

    عائم     reg[el_w];
    ثابت صحيح lane = sg.get_local_linear_id();

#pragma unroll
    لكل (صحيح i = 0; i < el_w; ++i) {
        reg[i] = static_cast<عائم>(src[i * WARP_SIZE + lane]) * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#pragma unroll
    لكل (صحيح h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        لكل (صحيح j = 0; j < el_w; ++j) {
            ثابت عائم val  = reg[j];
            ثابت عائم val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#pragma unroll
    لكل (صحيح h = WARP_SIZE; h < N; h *= 2) {
        ثابت صحيح step = h / WARP_SIZE;
#pragma unroll
        لكل (صحيح j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            لكل (صحيح k = 0; k < step; ++k) {
                ثابت عائم x = reg[j + k];
                ثابت عائم y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    لكل (صحيح i = 0; i < el_w; ++i) {
        dst[i * WARP_SIZE + lane] = reg[i];
    }
}

template <صحيح N, typename T>
ساكن فراغ launch_fwht(ثابت T * src, عائم * dst, ثابت int64_t n_rows, ثابت عائم scale,
                        dpct::queue_ptr stream) {
    constexpr صحيح rows_per_block = 4;

    ثابت int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ثابت sycl::range<2> global(num_blocks * rows_per_block, WARP_SIZE);
    ثابت sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel<N, T>(src, dst, n_rows, scale, item);
                         });
}

// Wide blocks: one row per work-group instead of per sub-group, so each work-item
// keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
// still shuffle; those up to NT go through work-group local memory; the rest stay
// in registers.
template <صحيح N, صحيح NT, typename T>
ساكن فراغ fwht_kernel_wide(ثابت T * __restrict__ src,
                             عائم * __restrict__ dst,
                             ثابت int64_t            n_rows,
                             ثابت عائم              scale,
                             ثابت sycl::nd_item<2> & item,
                             عائم *                  smem) {
    ثابت int64_t r = item.get_global_id(0);
    ܐܢ (r >= n_rows) {
        ܦܢܐ;
    }

    src += r * N;
    dst += r * N;

    constexpr صحيح el_w = N / NT;
    static_assert(el_w >= 1 && N % NT == 0, "row must be a whole number of work-group widths");

    ثابت صحيح tid = item.get_local_id(1);

    عائم reg[el_w];
#pragma unroll
    لكل (صحيح i = 0; i < el_w; ++i) {
        reg[i] = static_cast<عائم>(src[i * NT + tid]) * scale;
    }

    ثابت sycl::sub_group sg   = item.get_sub_group();
    ثابت صحيح             lane = sg.get_local_linear_id();

    // Butterflies inside the sub-group, same pattern as the narrow kernel.
#pragma unroll
    لكل (صحيح h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        لكل (صحيح j = 0; j < el_w; ++j) {
            ثابت عائم val  = reg[j];
            ثابت عائم val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies from the sub-group width up to NT: the partner lane is outside
    // this sub-group, so it goes through work-group local memory instead of a shuffle.
    لكل (صحيح h = WARP_SIZE; h < NT; h *= 2) {
#pragma unroll
        لكل (صحيح j = 0; j < el_w; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        لكل (صحيح j = 0; j < el_w; ++j) {
            ثابت عائم val  = reg[j];
            ثابت عائم val2 = smem[j * NT + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    // Butterflies across registers: h is a multiple of NT, so the partner of element
    // i*NT + tid lives in reg[i + h/NT] on the same work-item.
    لكل (صحيح h = NT; h < N; h *= 2) {
        ثابت صحيح step = h / NT;
        لكل (صحيح j = 0; j < el_w; j += 2 * step) {
            لكل (صحيح k = 0; k < step; ++k) {
                ثابت عائم x = reg[j + k];
                ثابت عائم y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    لكل (صحيح i = 0; i < el_w; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

template <صحيح N, صحيح NT, typename T>
ساكن فراغ launch_fwht_wide(ثابت T *       src,
                             عائم *         dst,
                             ثابت int64_t   n_rows,
                             ثابت عائم     scale,
                             dpct::queue_ptr stream) {
    ثابت sycl::range<2> global(n_rows, NT);
    ثابت sycl::range<2> local(1, NT);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<عائم, 1> smem(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel_wide<N, NT, T>(src, dst, n_rows, scale, item, get_pointer(smem));
                         });
    });
}

template <صحيح N, صحيح m, typename T>
ساكن فراغ kronecker_kernel(ثابت T * __restrict__ src,
                             عائم * __restrict__ dst,
                             ثابت int64_t            n_rows,
                             ثابت عائم              scale,
                             ثابت sycl::nd_item<2> & item) {
    static_assert(m == 12 || m == 20, "block size has to be 12 or 20.");

    ثابت sycl::sub_group sg = item.get_sub_group();

    ثابت int64_t r = item.get_global_id(0);
    ܐܢ (r >= n_rows) {
        ܦܢܐ;
    }

    src += r * N;
    dst += r * N;

    constexpr صحيح blocks_per_group = N / m;
    constexpr صحيح el_w             = blocks_per_group / WARP_SIZE;
    static_assert(el_w >= 1 && blocks_per_group % WARP_SIZE == 0, "blocks_per_group must be a multiple of WARP_SIZE");
    عائم     reg[el_w * m];
    ثابت صحيح lane = sg.get_local_linear_id();

#pragma unroll
    لكل (صحيح i = 0; i < el_w; ++i) {
        ثابت صحيح b_idx = i * WARP_SIZE + lane;

#pragma unroll
        لكل (صحيح j = 0; j < m; ++j) {
            reg[i * m + j] = static_cast<عائم>(src[b_idx * m + j]) * scale;
        }
    }

#pragma unroll
    لكل (صحيح b = 0; b < el_w; ++b) {
        عائم z[m] = { 0.0f };

#pragma unroll
        لكل (صحيح i = 0; i < m; ++i) {
#pragma unroll
            لكل (صحيح j = 0; j < m; ++j) {
                ثابت عائم h = (m == 12 ? H12[j][i] : H20[j][i]);
                z[i] += reg[b * m + j] * h;
            }
        }

#pragma unroll
        لكل (صحيح i = 0; i < m; ++i) {
            reg[b * m + i] = z[i];
        }
    }

#pragma unroll
    لكل (صحيح h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        لكل (صحيح j = 0; j < el_w; ++j) {
#pragma unroll
            لكل (صحيح k = 0; k < m; ++k) {
                ثابت عائم val  = reg[j * m + k];
                ثابت عائم val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

                reg[j * m + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#pragma unroll
    لكل (صحيح h = WARP_SIZE; h < blocks_per_group; h *= 2) {
        ثابت صحيح step = h / WARP_SIZE;
#pragma unroll
        لكل (صحيح j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            لكل (صحيح s = 0; s < step; ++s) {
#pragma unroll
                لكل (صحيح k = 0; k < m; ++k) {
                    ثابت عائم x = reg[(j + s) * m + k];
                    ثابت عائم y = reg[(j + s + step) * m + k];

                    reg[(j + s) * m + k]        = x + y;
                    reg[(j + s + step) * m + k] = x - y;
                }
            }
        }
    }

#pragma unroll
    لكل (صحيح i = 0; i < el_w; ++i) {
        ثابت صحيح b_idx = i * WARP_SIZE + lane;
#pragma unroll
        لكل (صحيح k = 0; k < m; ++k) {
            dst[b_idx * m + k] = reg[i * m + k];
        }
    }
}

template <صحيح N, صحيح m, typename T>
ساكن فراغ launch_kronecker(ثابت T *       src,
                             عائم *         dst,
                             ثابت int64_t   n_rows,
                             ثابت عائم     scale,
                             dpct::queue_ptr stream) {
    constexpr صحيح rows_per_block = 4;

    ثابت int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ثابت sycl::range<2> global(num_blocks * rows_per_block, WARP_SIZE);
    ثابت sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             kronecker_kernel<N, m, T>(src, dst, n_rows, scale, item);
                         });
}

template <typename T>
ساكن منطقي ggml_sycl_op_fwht_impl(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src, ggml_tensor * dst) {
    ثابت صحيح     n    = (صحيح) src->ne[0];
    ثابت int64_t rows = ggml_nrows(src);

    ثابت T *       src_d  = (ثابت T *) src->data;
    عائم *         dst_d  = (عائم *) dst->data;
    dpct::queue_ptr stream = ctx.stream();

    ثابت عائم scale = 1.0f / std::sqrt((عائم) n);

    بدل (n) {
        حالة 64:
            launch_fwht<64>(src_d, dst_d, rows, scale, stream);
            ܦܢܐ ܫܪܪܐ;
        حالة 128:
            launch_fwht<128>(src_d, dst_d, rows, scale, stream);
            ܦܢܐ ܫܪܪܐ;
        حالة 256:
            launch_fwht<256>(src_d, dst_d, rows, scale, stream);
            ܦܢܐ ܫܪܪܐ;
        حالة 512:
            launch_fwht<512>(src_d, dst_d, rows, scale, stream);
            ܦܢܐ ܫܪܪܐ;
        حالة 384:
            launch_kronecker<384, 12>(src_d, dst_d, rows, scale, stream);
            ܦܢܐ ܫܪܪܐ;
        حالة 768:
            launch_kronecker<768, 12>(src_d, dst_d, rows, scale, stream);
            ܦܢܐ ܫܪܪܐ;
        حالة 640:
            launch_kronecker<640, 20>(src_d, dst_d, rows, scale, stream);
            ܦܢܐ ܫܪܪܐ;
        حالة 1280:
            launch_kronecker<1280, 20>(src_d, dst_d, rows, scale, stream);
            ܦܢܐ ܫܪܪܐ;
        حالة 1024:
            launch_fwht_wide<1024, 256>(src_d, dst_d, rows, scale, stream);
            ܦܢܐ ܫܪܪܐ;
        حالة 2048:
            launch_fwht_wide<2048, 256>(src_d, dst_d, rows, scale, stream);
            ܦܢܐ ܫܪܪܐ;
        حالة 4096:
            launch_fwht_wide<4096, 256>(src_d, dst_d, rows, scale, stream);
            ܦܢܐ ܫܪܪܐ;
        حالة 8192:
            launch_fwht_wide<8192, 256>(src_d, dst_d, rows, scale, stream);
            ܦܢܐ ܫܪܪܐ;
        افتراضي:
            ܦܢܐ ܕܓܠܐ;
    }
}

منطقي ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src, ggml_tensor * dst) {
    ܐܢ (dst->type != GGML_TYPE_F32) {
        ܦܢܐ ܕܓܠܐ;
    }
    ܐܢ (!ggml_are_same_shape(src, dst)) {
        ܦܢܐ ܕܓܠܐ;
    }
    ܐܢ (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        ܦܢܐ ܕܓܠܐ;
    }

    بدل (src->type) {
        حالة GGML_TYPE_F32:
            ܦܢܐ ggml_sycl_op_fwht_impl<عائم>(ctx, src, dst);
        حالة GGML_TYPE_F16:
            ܦܢܐ ggml_sycl_op_fwht_impl<sycl::half>(ctx, src, dst);
        افتراضي:
            ܦܢܐ ܕܓܠܐ;
    }
}
