#অন্তর্ভাব "fwht.hpp"

#অন্তর্ভাব <cmath>
#define P 1.0f
#define N -1.0f

// constant Hadamard matrix via Paley I construction
স্থির constexpr দশমিক H12[12][12] = {
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

স্থির constexpr দশমিক H20[20][20] = {
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

template <পূর্ণসংখ্যা N, typename T>
স্থির শূন্য fwht_kernel(ধ্রুবক T * __restrict__ src, দশমিক * __restrict__ dst, ধ্রুবক int64_t n_rows,
                        ধ্রুবক দশমিক scale, ধ্রুবক sycl::nd_item<2> & item) {
    ধ্রুবক sycl::sub_group sg = item.get_sub_group();

    ধ্রুবক int64_t r = item.get_global_id(0);
    যদি (r >= n_rows) {
        ফেরত;
    }

    src += r * N;
    dst += r * N;

    constexpr পূর্ণসংখ্যা el_w = N / WARP_আকার;
    static_assert(el_w >= 1 && N % WARP_আকার == 0, "row must be a whole number of sub-group widths");

    দশমিক     reg[el_w];
    ধ্রুবক পূর্ণসংখ্যা lane = sg.get_local_linear_id();

#pragma unroll
    জন্য (পূর্ণসংখ্যা i = 0; i < el_w; ++i) {
        reg[i] = static_cast<দশমিক>(src[i * WARP_আকার + lane]) * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#pragma unroll
    জন্য (পূর্ণসংখ্যা h = 1; h < WARP_আকার; h *= 2) {
#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < el_w; ++j) {
            ধ্রুবক দশমিক val  = reg[j];
            ধ্রুবক দশমিক val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_আকার);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#pragma unroll
    জন্য (পূর্ণসংখ্যা h = WARP_আকার; h < N; h *= 2) {
        ধ্রুবক পূর্ণসংখ্যা step = h / WARP_আকার;
#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            জন্য (পূর্ণসংখ্যা k = 0; k < step; ++k) {
                ধ্রুবক দশমিক x = reg[j + k];
                ধ্রুবক দশমিক y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    জন্য (পূর্ণসংখ্যা i = 0; i < el_w; ++i) {
        dst[i * WARP_আকার + lane] = reg[i];
    }
}

template <পূর্ণসংখ্যা N, typename T>
স্থির শূন্য launch_fwht(ধ্রুবক T * src, দশমিক * dst, ধ্রুবক int64_t n_rows, ধ্রুবক দশমিক scale,
                        dpct::queue_ptr stream) {
    constexpr পূর্ণসংখ্যা rows_per_block = 4;

    ধ্রুবক int64_t অঙ্ক_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ধ্রুবক sycl::range<2> global(অঙ্ক_blocks * rows_per_block, WARP_আকার);
    ধ্রুবক sycl::range<2> local(rows_per_block, WARP_আকার);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                             fwht_kernel<N, T>(src, dst, n_rows, scale, item);
                         });
}

// Wide blocks: one row per work-group instead of per sub-group, so each work-item
// keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
// still shuffle; those up to NT go through work-group local memory; the rest stay
// in registers.
template <পূর্ণসংখ্যা N, পূর্ণসংখ্যা NT, typename T>
স্থির শূন্য fwht_kernel_wide(ধ্রুবক T * __restrict__ src,
                             দশমিক * __restrict__ dst,
                             ধ্রুবক int64_t            n_rows,
                             ধ্রুবক দশমিক              scale,
                             ধ্রুবক sycl::nd_item<2> & item,
                             দশমিক *                  smem) {
    ধ্রুবক int64_t r = item.get_global_id(0);
    যদি (r >= n_rows) {
        ফেরত;
    }

    src += r * N;
    dst += r * N;

    constexpr পূর্ণসংখ্যা el_w = N / NT;
    static_assert(el_w >= 1 && N % NT == 0, "row must be a whole number of work-group widths");

    ধ্রুবক পূর্ণসংখ্যা tid = item.get_local_id(1);

    দশমিক reg[el_w];
#pragma unroll
    জন্য (পূর্ণসংখ্যা i = 0; i < el_w; ++i) {
        reg[i] = static_cast<দশমিক>(src[i * NT + tid]) * scale;
    }

    ধ্রুবক sycl::sub_group sg   = item.get_sub_group();
    ধ্রুবক পূর্ণসংখ্যা             lane = sg.get_local_linear_id();

    // Butterflies inside the sub-group, same pattern as the narrow kernel.
#pragma unroll
    জন্য (পূর্ণসংখ্যা h = 1; h < WARP_আকার; h *= 2) {
#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < el_w; ++j) {
            ধ্রুবক দশমিক val  = reg[j];
            ধ্রুবক দশমিক val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_আকার);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies from the sub-group width up to NT: the partner lane is outside
    // this sub-group, so it goes through work-group local memory instead of a shuffle.
    জন্য (পূর্ণসংখ্যা h = WARP_আকার; h < NT; h *= 2) {
#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < el_w; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < el_w; ++j) {
            ধ্রুবক দশমিক val  = reg[j];
            ধ্রুবক দশমিক val2 = smem[j * NT + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    // Butterflies across registers: h is a multiple of NT, so the partner of element
    // i*NT + tid lives in reg[i + h/NT] on the same work-item.
    জন্য (পূর্ণসংখ্যা h = NT; h < N; h *= 2) {
        ধ্রুবক পূর্ণসংখ্যা step = h / NT;
        জন্য (পূর্ণসংখ্যা j = 0; j < el_w; j += 2 * step) {
            জন্য (পূর্ণসংখ্যা k = 0; k < step; ++k) {
                ধ্রুবক দশমিক x = reg[j + k];
                ধ্রুবক দশমিক y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    জন্য (পূর্ণসংখ্যা i = 0; i < el_w; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

template <পূর্ণসংখ্যা N, পূর্ণসংখ্যা NT, typename T>
স্থির শূন্য launch_fwht_wide(ধ্রুবক T *       src,
                             দশমিক *         dst,
                             ধ্রুবক int64_t   n_rows,
                             ধ্রুবক দশমিক     scale,
                             dpct::queue_ptr stream) {
    ধ্রুবক sycl::range<2> global(n_rows, NT);
    ধ্রুবক sycl::range<2> local(1, NT);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<দশমিক, 1> smem(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                             fwht_kernel_wide<N, NT, T>(src, dst, n_rows, scale, item, get_pointer(smem));
                         });
    });
}

template <পূর্ণসংখ্যা N, পূর্ণসংখ্যা m, typename T>
স্থির শূন্য kronecker_kernel(ধ্রুবক T * __restrict__ src,
                             দশমিক * __restrict__ dst,
                             ধ্রুবক int64_t            n_rows,
                             ধ্রুবক দশমিক              scale,
                             ধ্রুবক sycl::nd_item<2> & item) {
    static_assert(m == 12 || m == 20, "block size has to be 12 or 20.");

    ধ্রুবক sycl::sub_group sg = item.get_sub_group();

    ধ্রুবক int64_t r = item.get_global_id(0);
    যদি (r >= n_rows) {
        ফেরত;
    }

    src += r * N;
    dst += r * N;

    constexpr পূর্ণসংখ্যা blocks_per_group = N / m;
    constexpr পূর্ণসংখ্যা el_w             = blocks_per_group / WARP_আকার;
    static_assert(el_w >= 1 && blocks_per_group % WARP_আকার == 0, "blocks_per_group must be a multiple of WARP_SIZE");
    দশমিক     reg[el_w * m];
    ধ্রুবক পূর্ণসংখ্যা lane = sg.get_local_linear_id();

#pragma unroll
    জন্য (পূর্ণসংখ্যা i = 0; i < el_w; ++i) {
        ধ্রুবক পূর্ণসংখ্যা b_idx = i * WARP_আকার + lane;

#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < m; ++j) {
            reg[i * m + j] = static_cast<দশমিক>(src[b_idx * m + j]) * scale;
        }
    }

#pragma unroll
    জন্য (পূর্ণসংখ্যা b = 0; b < el_w; ++b) {
        দশমিক z[m] = { 0.0f };

#pragma unroll
        জন্য (পূর্ণসংখ্যা i = 0; i < m; ++i) {
#pragma unroll
            জন্য (পূর্ণসংখ্যা j = 0; j < m; ++j) {
                ধ্রুবক দশমিক h = (m == 12 ? H12[j][i] : H20[j][i]);
                z[i] += reg[b * m + j] * h;
            }
        }

#pragma unroll
        জন্য (পূর্ণসংখ্যা i = 0; i < m; ++i) {
            reg[b * m + i] = z[i];
        }
    }

#pragma unroll
    জন্য (পূর্ণসংখ্যা h = 1; h < WARP_আকার; h *= 2) {
#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < el_w; ++j) {
#pragma unroll
            জন্য (পূর্ণসংখ্যা k = 0; k < m; ++k) {
                ধ্রুবক দশমিক val  = reg[j * m + k];
                ধ্রুবক দশমিক val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_আকার);

                reg[j * m + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#pragma unroll
    জন্য (পূর্ণসংখ্যা h = WARP_আকার; h < blocks_per_group; h *= 2) {
        ধ্রুবক পূর্ণসংখ্যা step = h / WARP_আকার;
#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            জন্য (পূর্ণসংখ্যা s = 0; s < step; ++s) {
#pragma unroll
                জন্য (পূর্ণসংখ্যা k = 0; k < m; ++k) {
                    ধ্রুবক দশমিক x = reg[(j + s) * m + k];
                    ধ্রুবক দশমিক y = reg[(j + s + step) * m + k];

                    reg[(j + s) * m + k]        = x + y;
                    reg[(j + s + step) * m + k] = x - y;
                }
            }
        }
    }

#pragma unroll
    জন্য (পূর্ণসংখ্যা i = 0; i < el_w; ++i) {
        ধ্রুবক পূর্ণসংখ্যা b_idx = i * WARP_আকার + lane;
#pragma unroll
        জন্য (পূর্ণসংখ্যা k = 0; k < m; ++k) {
            dst[b_idx * m + k] = reg[i * m + k];
        }
    }
}

template <পূর্ণসংখ্যা N, পূর্ণসংখ্যা m, typename T>
স্থির শূন্য launch_kronecker(ধ্রুবক T *       src,
                             দশমিক *         dst,
                             ধ্রুবক int64_t   n_rows,
                             ধ্রুবক দশমিক     scale,
                             dpct::queue_ptr stream) {
    constexpr পূর্ণসংখ্যা rows_per_block = 4;

    ধ্রুবক int64_t অঙ্ক_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ধ্রুবক sycl::range<2> global(অঙ্ক_blocks * rows_per_block, WARP_আকার);
    ধ্রুবক sycl::range<2> local(rows_per_block, WARP_আকার);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                             kronecker_kernel<N, m, T>(src, dst, n_rows, scale, item);
                         });
}

template <typename T>
স্থির যৌক্তিক ggml_sycl_op_fwht_impl(ggml_backend_sycl_context & ctx, ধ্রুবক ggml_tensor * src, ggml_tensor * dst) {
    ধ্রুবক পূর্ণসংখ্যা     n    = (পূর্ণসংখ্যা) src->ne[0];
    ধ্রুবক int64_t rows = ggml_nrows(src);

    ধ্রুবক T *       src_d  = (ধ্রুবক T *) src->তথ্য;
    দশমিক *         dst_d  = (দশমিক *) dst->তথ্য;
    dpct::queue_ptr stream = ctx.stream();

    ধ্রুবক দশমিক scale = 1.0f / std::sqrt((দশমিক) n);

    নির্বাচন (n) {
        ক্ষেত্র 64:
            launch_fwht<64>(src_d, dst_d, rows, scale, stream);
            ফেরত সত্য;
        ক্ষেত্র 128:
            launch_fwht<128>(src_d, dst_d, rows, scale, stream);
            ফেরত সত্য;
        ক্ষেত্র 256:
            launch_fwht<256>(src_d, dst_d, rows, scale, stream);
            ফেরত সত্য;
        ক্ষেত্র 512:
            launch_fwht<512>(src_d, dst_d, rows, scale, stream);
            ফেরত সত্য;
        ক্ষেত্র 384:
            launch_kronecker<384, 12>(src_d, dst_d, rows, scale, stream);
            ফেরত সত্য;
        ক্ষেত্র 768:
            launch_kronecker<768, 12>(src_d, dst_d, rows, scale, stream);
            ফেরত সত্য;
        ক্ষেত্র 640:
            launch_kronecker<640, 20>(src_d, dst_d, rows, scale, stream);
            ফেরত সত্য;
        ক্ষেত্র 1280:
            launch_kronecker<1280, 20>(src_d, dst_d, rows, scale, stream);
            ফেরত সত্য;
        ক্ষেত্র 1024:
            launch_fwht_wide<1024, 256>(src_d, dst_d, rows, scale, stream);
            ফেরত সত্য;
        ক্ষেত্র 2048:
            launch_fwht_wide<2048, 256>(src_d, dst_d, rows, scale, stream);
            ফেরত সত্য;
        ক্ষেত্র 4096:
            launch_fwht_wide<4096, 256>(src_d, dst_d, rows, scale, stream);
            ফেরত সত্য;
        ক্ষেত্র 8192:
            launch_fwht_wide<8192, 256>(src_d, dst_d, rows, scale, stream);
            ফেরত সত্য;
        পূর্বনির্ধারিত:
            ফেরত মিথ্যা;
    }
}

যৌক্তিক ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, ধ্রুবক ggml_tensor * src, ggml_tensor * dst) {
    যদি (dst->type != GGML_TYPE_F32) {
        ফেরত মিথ্যা;
    }
    যদি (!ggml_are_same_shape(src, dst)) {
        ফেরত মিথ্যা;
    }
    যদি (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        ফেরত মিথ্যা;
    }

    নির্বাচন (src->type) {
        ক্ষেত্র GGML_TYPE_F32:
            ফেরত ggml_sycl_op_fwht_impl<দশমিক>(ctx, src, dst);
        ক্ষেত্র GGML_TYPE_F16:
            ফেরত ggml_sycl_op_fwht_impl<sycl::half>(ctx, src, dst);
        পূর্বনির্ধারিত:
            ফেরত মিথ্যা;
    }
}
