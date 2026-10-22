#অন্তর্ভাৱ "fwht.hpp"

#অন্তর্ভাৱ <cmath>
#define P 1.0f
#define N -1.0f

// constant Hadamard matrix via Paley I construction
স্থির constexpr ভিন্ন H12[12][12] = {
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

স্থির constexpr ভিন্ন H20[20][20] = {
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

template <পূর্ণাঙ্ক N, typename T>
স্থির শূন্য fwht_kernel(ধ্রুৱ T * __restrict__ src, ভিন্ন * __restrict__ dst, ধ্রুৱ int64_t n_rows,
                        ধ্রুৱ ভিন্ন scale, ধ্রুৱ sycl::nd_item<2> & item) {
    ধ্রুৱ sycl::sub_group sg = item.get_sub_group();

    ধ্রুৱ int64_t r = item.get_global_id(0);
    যদি (r >= n_rows) {
        ঘূৰাই;
    }

    src += r * N;
    dst += r * N;

    constexpr পূর্ণাঙ্ক el_w = N / WARP_SIZE;
    static_assert(el_w >= 1 && N % WARP_SIZE == 0, "row must be a whole number of sub-group widths");

    ভিন্ন     reg[el_w];
    ধ্রুৱ পূর্ণাঙ্ক lane = sg.get_local_linear_id();

#pragma unroll
    বাবে (পূর্ণাঙ্ক i = 0; i < el_w; ++i) {
        reg[i] = static_cast<ভিন্ন>(src[i * WARP_SIZE + lane]) * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#pragma unroll
    বাবে (পূর্ণাঙ্ক h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        বাবে (পূর্ণাঙ্ক j = 0; j < el_w; ++j) {
            ধ্রুৱ ভিন্ন val  = reg[j];
            ধ্রুৱ ভিন্ন val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#pragma unroll
    বাবে (পূর্ণাঙ্ক h = WARP_SIZE; h < N; h *= 2) {
        ধ্রুৱ পূর্ণাঙ্ক step = h / WARP_SIZE;
#pragma unroll
        বাবে (পূর্ণাঙ্ক j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            বাবে (পূর্ণাঙ্ক k = 0; k < step; ++k) {
                ধ্রুৱ ভিন্ন x = reg[j + k];
                ধ্রুৱ ভিন্ন y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    বাবে (পূর্ণাঙ্ক i = 0; i < el_w; ++i) {
        dst[i * WARP_SIZE + lane] = reg[i];
    }
}

template <পূর্ণাঙ্ক N, typename T>
স্থির শূন্য launch_fwht(ধ্রুৱ T * src, ভিন্ন * dst, ধ্রুৱ int64_t n_rows, ধ্রুৱ ভিন্ন scale,
                        dpct::queue_ptr stream) {
    constexpr পূর্ণাঙ্ক rows_per_block = 4;

    ধ্রুৱ int64_t অঙ্ক_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ধ্রুৱ sycl::range<2> global(অঙ্ক_blocks * rows_per_block, WARP_SIZE);
    ধ্রুৱ sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel<N, T>(src, dst, n_rows, scale, item);
                         });
}

// Wide blocks: one row per work-group instead of per sub-group, so each work-item
// keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
// still shuffle; those up to NT go through work-group local memory; the rest stay
// in registers.
template <পূর্ণাঙ্ক N, পূর্ণাঙ্ক NT, typename T>
স্থির শূন্য fwht_kernel_wide(ধ্রুৱ T * __restrict__ src,
                             ভিন্ন * __restrict__ dst,
                             ধ্রুৱ int64_t            n_rows,
                             ধ্রুৱ ভিন্ন              scale,
                             ধ্রুৱ sycl::nd_item<2> & item,
                             ভিন্ন *                  smem) {
    ধ্রুৱ int64_t r = item.get_global_id(0);
    যদি (r >= n_rows) {
        ঘূৰাই;
    }

    src += r * N;
    dst += r * N;

    constexpr পূর্ণাঙ্ক el_w = N / NT;
    static_assert(el_w >= 1 && N % NT == 0, "row must be a whole number of work-group widths");

    ধ্রুৱ পূর্ণাঙ্ক tid = item.get_local_id(1);

    ভিন্ন reg[el_w];
#pragma unroll
    বাবে (পূর্ণাঙ্ক i = 0; i < el_w; ++i) {
        reg[i] = static_cast<ভিন্ন>(src[i * NT + tid]) * scale;
    }

    ধ্রুৱ sycl::sub_group sg   = item.get_sub_group();
    ধ্রুৱ পূর্ণাঙ্ক             lane = sg.get_local_linear_id();

    // Butterflies inside the sub-group, same pattern as the narrow kernel.
#pragma unroll
    বাবে (পূর্ণাঙ্ক h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        বাবে (পূর্ণাঙ্ক j = 0; j < el_w; ++j) {
            ধ্রুৱ ভিন্ন val  = reg[j];
            ধ্রুৱ ভিন্ন val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies from the sub-group width up to NT: the partner lane is outside
    // this sub-group, so it goes through work-group local memory instead of a shuffle.
    বাবে (পূর্ণাঙ্ক h = WARP_SIZE; h < NT; h *= 2) {
#pragma unroll
        বাবে (পূর্ণাঙ্ক j = 0; j < el_w; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        বাবে (পূর্ণাঙ্ক j = 0; j < el_w; ++j) {
            ধ্রুৱ ভিন্ন val  = reg[j];
            ধ্রুৱ ভিন্ন val2 = smem[j * NT + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    // Butterflies across registers: h is a multiple of NT, so the partner of element
    // i*NT + tid lives in reg[i + h/NT] on the same work-item.
    বাবে (পূর্ণাঙ্ক h = NT; h < N; h *= 2) {
        ধ্রুৱ পূর্ণাঙ্ক step = h / NT;
        বাবে (পূর্ণাঙ্ক j = 0; j < el_w; j += 2 * step) {
            বাবে (পূর্ণাঙ্ক k = 0; k < step; ++k) {
                ধ্রুৱ ভিন্ন x = reg[j + k];
                ধ্রুৱ ভিন্ন y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    বাবে (পূর্ণাঙ্ক i = 0; i < el_w; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

template <পূর্ণাঙ্ক N, পূর্ণাঙ্ক NT, typename T>
স্থির শূন্য launch_fwht_wide(ধ্রুৱ T *       src,
                             ভিন্ন *         dst,
                             ধ্রুৱ int64_t   n_rows,
                             ধ্রুৱ ভিন্ন     scale,
                             dpct::queue_ptr stream) {
    ধ্রুৱ sycl::range<2> global(n_rows, NT);
    ধ্রুৱ sycl::range<2> local(1, NT);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<ভিন্ন, 1> smem(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel_wide<N, NT, T>(src, dst, n_rows, scale, item, get_pointer(smem));
                         });
    });
}

template <পূর্ণাঙ্ক N, পূর্ণাঙ্ক m, typename T>
স্থির শূন্য kronecker_kernel(ধ্রুৱ T * __restrict__ src,
                             ভিন্ন * __restrict__ dst,
                             ধ্রুৱ int64_t            n_rows,
                             ধ্রুৱ ভিন্ন              scale,
                             ধ্রুৱ sycl::nd_item<2> & item) {
    static_assert(m == 12 || m == 20, "block size has to be 12 or 20.");

    ধ্রুৱ sycl::sub_group sg = item.get_sub_group();

    ধ্রুৱ int64_t r = item.get_global_id(0);
    যদি (r >= n_rows) {
        ঘূৰাই;
    }

    src += r * N;
    dst += r * N;

    constexpr পূর্ণাঙ্ক blocks_per_group = N / m;
    constexpr পূর্ণাঙ্ক el_w             = blocks_per_group / WARP_SIZE;
    static_assert(el_w >= 1 && blocks_per_group % WARP_SIZE == 0, "blocks_per_group must be a multiple of WARP_SIZE");
    ভিন্ন     reg[el_w * m];
    ধ্রুৱ পূর্ণাঙ্ক lane = sg.get_local_linear_id();

#pragma unroll
    বাবে (পূর্ণাঙ্ক i = 0; i < el_w; ++i) {
        ধ্রুৱ পূর্ণাঙ্ক b_idx = i * WARP_SIZE + lane;

#pragma unroll
        বাবে (পূর্ণাঙ্ক j = 0; j < m; ++j) {
            reg[i * m + j] = static_cast<ভিন্ন>(src[b_idx * m + j]) * scale;
        }
    }

#pragma unroll
    বাবে (পূর্ণাঙ্ক b = 0; b < el_w; ++b) {
        ভিন্ন z[m] = { 0.0f };

#pragma unroll
        বাবে (পূর্ণাঙ্ক i = 0; i < m; ++i) {
#pragma unroll
            বাবে (পূর্ণাঙ্ক j = 0; j < m; ++j) {
                ধ্রুৱ ভিন্ন h = (m == 12 ? H12[j][i] : H20[j][i]);
                z[i] += reg[b * m + j] * h;
            }
        }

#pragma unroll
        বাবে (পূর্ণাঙ্ক i = 0; i < m; ++i) {
            reg[b * m + i] = z[i];
        }
    }

#pragma unroll
    বাবে (পূর্ণাঙ্ক h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        বাবে (পূর্ণাঙ্ক j = 0; j < el_w; ++j) {
#pragma unroll
            বাবে (পূর্ণাঙ্ক k = 0; k < m; ++k) {
                ধ্রুৱ ভিন্ন val  = reg[j * m + k];
                ধ্রুৱ ভিন্ন val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

                reg[j * m + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#pragma unroll
    বাবে (পূর্ণাঙ্ক h = WARP_SIZE; h < blocks_per_group; h *= 2) {
        ধ্রুৱ পূর্ণাঙ্ক step = h / WARP_SIZE;
#pragma unroll
        বাবে (পূর্ণাঙ্ক j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            বাবে (পূর্ণাঙ্ক s = 0; s < step; ++s) {
#pragma unroll
                বাবে (পূর্ণাঙ্ক k = 0; k < m; ++k) {
                    ধ্রুৱ ভিন্ন x = reg[(j + s) * m + k];
                    ধ্রুৱ ভিন্ন y = reg[(j + s + step) * m + k];

                    reg[(j + s) * m + k]        = x + y;
                    reg[(j + s + step) * m + k] = x - y;
                }
            }
        }
    }

#pragma unroll
    বাবে (পূর্ণাঙ্ক i = 0; i < el_w; ++i) {
        ধ্রুৱ পূর্ণাঙ্ক b_idx = i * WARP_SIZE + lane;
#pragma unroll
        বাবে (পূর্ণাঙ্ক k = 0; k < m; ++k) {
            dst[b_idx * m + k] = reg[i * m + k];
        }
    }
}

template <পূর্ণাঙ্ক N, পূর্ণাঙ্ক m, typename T>
স্থির শূন্য launch_kronecker(ধ্রুৱ T *       src,
                             ভিন্ন *         dst,
                             ধ্রুৱ int64_t   n_rows,
                             ধ্রুৱ ভিন্ন     scale,
                             dpct::queue_ptr stream) {
    constexpr পূর্ণাঙ্ক rows_per_block = 4;

    ধ্রুৱ int64_t অঙ্ক_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ধ্রুৱ sycl::range<2> global(অঙ্ক_blocks * rows_per_block, WARP_SIZE);
    ধ্রুৱ sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             kronecker_kernel<N, m, T>(src, dst, n_rows, scale, item);
                         });
}

template <typename T>
স্থির তার্কিক ggml_sycl_op_fwht_impl(ggml_backend_sycl_context & ctx, ধ্রুৱ ggml_tensor * src, ggml_tensor * dst) {
    ধ্রুৱ পূর্ণাঙ্ক     n    = (পূর্ণাঙ্ক) src->ne[0];
    ধ্রুৱ int64_t rows = ggml_nrows(src);

    ধ্রুৱ T *       src_d  = (ধ্রুৱ T *) src->data;
    ভিন্ন *         dst_d  = (ভিন্ন *) dst->data;
    dpct::queue_ptr stream = ctx.stream();

    ধ্রুৱ ভিন্ন scale = 1.0f / std::sqrt((ভিন্ন) n);

    ৱিকল্প (n) {
        পক্ষ 64:
            launch_fwht<64>(src_d, dst_d, rows, scale, stream);
            ঘূৰাই সঁচা;
        পক্ষ 128:
            launch_fwht<128>(src_d, dst_d, rows, scale, stream);
            ঘূৰাই সঁচা;
        পক্ষ 256:
            launch_fwht<256>(src_d, dst_d, rows, scale, stream);
            ঘূৰাই সঁচা;
        পক্ষ 512:
            launch_fwht<512>(src_d, dst_d, rows, scale, stream);
            ঘূৰাই সঁচা;
        পক্ষ 384:
            launch_kronecker<384, 12>(src_d, dst_d, rows, scale, stream);
            ঘূৰাই সঁচা;
        পক্ষ 768:
            launch_kronecker<768, 12>(src_d, dst_d, rows, scale, stream);
            ঘূৰাই সঁচা;
        পক্ষ 640:
            launch_kronecker<640, 20>(src_d, dst_d, rows, scale, stream);
            ঘূৰাই সঁচা;
        পক্ষ 1280:
            launch_kronecker<1280, 20>(src_d, dst_d, rows, scale, stream);
            ঘূৰাই সঁচা;
        পক্ষ 1024:
            launch_fwht_wide<1024, 256>(src_d, dst_d, rows, scale, stream);
            ঘূৰাই সঁচা;
        পক্ষ 2048:
            launch_fwht_wide<2048, 256>(src_d, dst_d, rows, scale, stream);
            ঘূৰাই সঁচা;
        পক্ষ 4096:
            launch_fwht_wide<4096, 256>(src_d, dst_d, rows, scale, stream);
            ঘূৰাই সঁচা;
        পক্ষ 8192:
            launch_fwht_wide<8192, 256>(src_d, dst_d, rows, scale, stream);
            ঘূৰাই সঁচা;
        স্ৱভাৱ:
            ঘূৰাই মিছা;
    }
}

তার্কিক ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, ধ্রুৱ ggml_tensor * src, ggml_tensor * dst) {
    যদি (dst->type != GGML_TYPE_F32) {
        ঘূৰাই মিছা;
    }
    যদি (!ggml_are_same_shape(src, dst)) {
        ঘূৰাই মিছা;
    }
    যদি (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        ঘূৰাই মিছা;
    }

    ৱিকল্প (src->type) {
        পক্ষ GGML_TYPE_F32:
            ঘূৰাই ggml_sycl_op_fwht_impl<ভিন্ন>(ctx, src, dst);
        পক্ষ GGML_TYPE_F16:
            ঘূৰাই ggml_sycl_op_fwht_impl<sycl::half>(ctx, src, dst);
        স্ৱভাৱ:
            ঘূৰাই মিছা;
    }
}
