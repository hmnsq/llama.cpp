#અન્તર્ભાવ "fwht.hpp"

#અન્તર્ભાવ <cmath>
#define P 1.0f
#define N -1.0f

// constant Hadamard matrix via Paley I construction
સ્થિર constexpr દશાંશ H12[12][12] = {
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

સ્થિર constexpr દશાંશ H20[20][20] = {
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

template <પૂર્ણાંક N, typename T>
સ્થિર શૂન્ય fwht_kernel(અચલ T * __restrict__ src, દશાંશ * __restrict__ dst, અચલ int64_t n_rows,
                        અચલ દશાંશ scale, અચલ sycl::nd_item<2> & item) {
    અચલ sycl::sub_group sg = item.get_sub_group();

    અચલ int64_t r = item.get_global_id(0);
    જો (r >= n_rows) {
        પાછા;
    }

    src += r * N;
    dst += r * N;

    constexpr પૂર્ણાંક el_w = N / WARP_SIZE;
    static_assert(el_w >= 1 && N % WARP_SIZE == 0, "row must be a whole number of sub-group widths");

    દશાંશ     reg[el_w];
    અચલ પૂર્ણાંક lane = sg.get_local_linear_id();

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < el_w; ++i) {
        reg[i] = static_cast<દશાંશ>(src[i * WARP_SIZE + lane]) * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#pragma unroll
    માટે (પૂર્ણાંક h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        માટે (પૂર્ણાંક j = 0; j < el_w; ++j) {
            અચલ દશાંશ val  = reg[j];
            અચલ દશાંશ val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#pragma unroll
    માટે (પૂર્ણાંક h = WARP_SIZE; h < N; h *= 2) {
        અચલ પૂર્ણાંક step = h / WARP_SIZE;
#pragma unroll
        માટે (પૂર્ણાંક j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            માટે (પૂર્ણાંક k = 0; k < step; ++k) {
                અચલ દશાંશ x = reg[j + k];
                અચલ દશાંશ y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < el_w; ++i) {
        dst[i * WARP_SIZE + lane] = reg[i];
    }
}

template <પૂર્ણાંક N, typename T>
સ્થિર શૂન્ય launch_fwht(અચલ T * src, દશાંશ * dst, અચલ int64_t n_rows, અચલ દશાંશ scale,
                        dpct::queue_ptr stream) {
    constexpr પૂર્ણાંક rows_per_block = 4;

    અચલ int64_t અંક_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    અચલ sycl::range<2> global(અંક_blocks * rows_per_block, WARP_SIZE);
    અચલ sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel<N, T>(src, dst, n_rows, scale, item);
                         });
}

// Wide blocks: one row per work-group instead of per sub-group, so each work-item
// keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
// still shuffle; those up to NT go through work-group local memory; the rest stay
// in registers.
template <પૂર્ણાંક N, પૂર્ણાંક NT, typename T>
સ્થિર શૂન્ય fwht_kernel_wide(અચલ T * __restrict__ src,
                             દશાંશ * __restrict__ dst,
                             અચલ int64_t            n_rows,
                             અચલ દશાંશ              scale,
                             અચલ sycl::nd_item<2> & item,
                             દશાંશ *                  smem) {
    અચલ int64_t r = item.get_global_id(0);
    જો (r >= n_rows) {
        પાછા;
    }

    src += r * N;
    dst += r * N;

    constexpr પૂર્ણાંક el_w = N / NT;
    static_assert(el_w >= 1 && N % NT == 0, "row must be a whole number of work-group widths");

    અચલ પૂર્ણાંક tid = item.get_local_id(1);

    દશાંશ reg[el_w];
#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < el_w; ++i) {
        reg[i] = static_cast<દશાંશ>(src[i * NT + tid]) * scale;
    }

    અચલ sycl::sub_group sg   = item.get_sub_group();
    અચલ પૂર્ણાંક             lane = sg.get_local_linear_id();

    // Butterflies inside the sub-group, same pattern as the narrow kernel.
#pragma unroll
    માટે (પૂર્ણાંક h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        માટે (પૂર્ણાંક j = 0; j < el_w; ++j) {
            અચલ દશાંશ val  = reg[j];
            અચલ દશાંશ val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies from the sub-group width up to NT: the partner lane is outside
    // this sub-group, so it goes through work-group local memory instead of a shuffle.
    માટે (પૂર્ણાંક h = WARP_SIZE; h < NT; h *= 2) {
#pragma unroll
        માટે (પૂર્ણાંક j = 0; j < el_w; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        માટે (પૂર્ણાંક j = 0; j < el_w; ++j) {
            અચલ દશાંશ val  = reg[j];
            અચલ દશાંશ val2 = smem[j * NT + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    // Butterflies across registers: h is a multiple of NT, so the partner of element
    // i*NT + tid lives in reg[i + h/NT] on the same work-item.
    માટે (પૂર્ણાંક h = NT; h < N; h *= 2) {
        અચલ પૂર્ણાંક step = h / NT;
        માટે (પૂર્ણાંક j = 0; j < el_w; j += 2 * step) {
            માટે (પૂર્ણાંક k = 0; k < step; ++k) {
                અચલ દશાંશ x = reg[j + k];
                અચલ દશાંશ y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < el_w; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

template <પૂર્ણાંક N, પૂર્ણાંક NT, typename T>
સ્થિર શૂન્ય launch_fwht_wide(અચલ T *       src,
                             દશાંશ *         dst,
                             અચલ int64_t   n_rows,
                             અચલ દશાંશ     scale,
                             dpct::queue_ptr stream) {
    અચલ sycl::range<2> global(n_rows, NT);
    અચલ sycl::range<2> local(1, NT);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<દશાંશ, 1> smem(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel_wide<N, NT, T>(src, dst, n_rows, scale, item, get_pointer(smem));
                         });
    });
}

template <પૂર્ણાંક N, પૂર્ણાંક m, typename T>
સ્થિર શૂન્ય kronecker_kernel(અચલ T * __restrict__ src,
                             દશાંશ * __restrict__ dst,
                             અચલ int64_t            n_rows,
                             અચલ દશાંશ              scale,
                             અચલ sycl::nd_item<2> & item) {
    static_assert(m == 12 || m == 20, "block size has to be 12 or 20.");

    અચલ sycl::sub_group sg = item.get_sub_group();

    અચલ int64_t r = item.get_global_id(0);
    જો (r >= n_rows) {
        પાછા;
    }

    src += r * N;
    dst += r * N;

    constexpr પૂર્ણાંક blocks_per_group = N / m;
    constexpr પૂર્ણાંક el_w             = blocks_per_group / WARP_SIZE;
    static_assert(el_w >= 1 && blocks_per_group % WARP_SIZE == 0, "blocks_per_group must be a multiple of WARP_SIZE");
    દશાંશ     reg[el_w * m];
    અચલ પૂર્ણાંક lane = sg.get_local_linear_id();

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < el_w; ++i) {
        અચલ પૂર્ણાંક b_idx = i * WARP_SIZE + lane;

#pragma unroll
        માટે (પૂર્ણાંક j = 0; j < m; ++j) {
            reg[i * m + j] = static_cast<દશાંશ>(src[b_idx * m + j]) * scale;
        }
    }

#pragma unroll
    માટે (પૂર્ણાંક b = 0; b < el_w; ++b) {
        દશાંશ z[m] = { 0.0f };

#pragma unroll
        માટે (પૂર્ણાંક i = 0; i < m; ++i) {
#pragma unroll
            માટે (પૂર્ણાંક j = 0; j < m; ++j) {
                અચલ દશાંશ h = (m == 12 ? H12[j][i] : H20[j][i]);
                z[i] += reg[b * m + j] * h;
            }
        }

#pragma unroll
        માટે (પૂર્ણાંક i = 0; i < m; ++i) {
            reg[b * m + i] = z[i];
        }
    }

#pragma unroll
    માટે (પૂર્ણાંક h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        માટે (પૂર્ણાંક j = 0; j < el_w; ++j) {
#pragma unroll
            માટે (પૂર્ણાંક k = 0; k < m; ++k) {
                અચલ દશાંશ val  = reg[j * m + k];
                અચલ દશાંશ val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

                reg[j * m + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#pragma unroll
    માટે (પૂર્ણાંક h = WARP_SIZE; h < blocks_per_group; h *= 2) {
        અચલ પૂર્ણાંક step = h / WARP_SIZE;
#pragma unroll
        માટે (પૂર્ણાંક j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            માટે (પૂર્ણાંક s = 0; s < step; ++s) {
#pragma unroll
                માટે (પૂર્ણાંક k = 0; k < m; ++k) {
                    અચલ દશાંશ x = reg[(j + s) * m + k];
                    અચલ દશાંશ y = reg[(j + s + step) * m + k];

                    reg[(j + s) * m + k]        = x + y;
                    reg[(j + s + step) * m + k] = x - y;
                }
            }
        }
    }

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < el_w; ++i) {
        અચલ પૂર્ણાંક b_idx = i * WARP_SIZE + lane;
#pragma unroll
        માટે (પૂર્ણાંક k = 0; k < m; ++k) {
            dst[b_idx * m + k] = reg[i * m + k];
        }
    }
}

template <પૂર્ણાંક N, પૂર્ણાંક m, typename T>
સ્થિર શૂન્ય launch_kronecker(અચલ T *       src,
                             દશાંશ *         dst,
                             અચલ int64_t   n_rows,
                             અચલ દશાંશ     scale,
                             dpct::queue_ptr stream) {
    constexpr પૂર્ણાંક rows_per_block = 4;

    અચલ int64_t અંક_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    અચલ sycl::range<2> global(અંક_blocks * rows_per_block, WARP_SIZE);
    અચલ sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             kronecker_kernel<N, m, T>(src, dst, n_rows, scale, item);
                         });
}

template <typename T>
સ્થિર તાર્કિક ggml_sycl_op_fwht_impl(ggml_backend_sycl_context & ctx, અચલ ggml_tensor * src, ggml_tensor * dst) {
    અચલ પૂર્ણાંક     n    = (પૂર્ણાંક) src->ne[0];
    અચલ int64_t rows = ggml_nrows(src);

    અચલ T *       src_d  = (અચલ T *) src->data;
    દશાંશ *         dst_d  = (દશાંશ *) dst->data;
    dpct::queue_ptr stream = ctx.stream();

    અચલ દશાંશ scale = 1.0f / std::sqrt((દશાંશ) n);

    પસંદગી (n) {
        સ્થિતિ 64:
            launch_fwht<64>(src_d, dst_d, rows, scale, stream);
            પાછા સાચું;
        સ્થિતિ 128:
            launch_fwht<128>(src_d, dst_d, rows, scale, stream);
            પાછા સાચું;
        સ્થિતિ 256:
            launch_fwht<256>(src_d, dst_d, rows, scale, stream);
            પાછા સાચું;
        સ્થિતિ 512:
            launch_fwht<512>(src_d, dst_d, rows, scale, stream);
            પાછા સાચું;
        સ્થિતિ 384:
            launch_kronecker<384, 12>(src_d, dst_d, rows, scale, stream);
            પાછા સાચું;
        સ્થિતિ 768:
            launch_kronecker<768, 12>(src_d, dst_d, rows, scale, stream);
            પાછા સાચું;
        સ્થિતિ 640:
            launch_kronecker<640, 20>(src_d, dst_d, rows, scale, stream);
            પાછા સાચું;
        સ્થિતિ 1280:
            launch_kronecker<1280, 20>(src_d, dst_d, rows, scale, stream);
            પાછા સાચું;
        સ્થિતિ 1024:
            launch_fwht_wide<1024, 256>(src_d, dst_d, rows, scale, stream);
            પાછા સાચું;
        સ્થિતિ 2048:
            launch_fwht_wide<2048, 256>(src_d, dst_d, rows, scale, stream);
            પાછા સાચું;
        સ્થિતિ 4096:
            launch_fwht_wide<4096, 256>(src_d, dst_d, rows, scale, stream);
            પાછા સાચું;
        સ્થિતિ 8192:
            launch_fwht_wide<8192, 256>(src_d, dst_d, rows, scale, stream);
            પાછા સાચું;
        મૂળભૂત:
            પાછા ખોટું;
    }
}

તાર્કિક ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, અચલ ggml_tensor * src, ggml_tensor * dst) {
    જો (dst->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }
    જો (!ggml_are_same_shape(src, dst)) {
        પાછા ખોટું;
    }
    જો (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        પાછા ખોટું;
    }

    પસંદગી (src->type) {
        સ્થિતિ GGML_TYPE_F32:
            પાછા ggml_sycl_op_fwht_impl<દશાંશ>(ctx, src, dst);
        સ્થિતિ GGML_TYPE_F16:
            પાછા ggml_sycl_op_fwht_impl<sycl::half>(ctx, src, dst);
        મૂળભૂત:
            પાછા ખોટું;
    }
}
