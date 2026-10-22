#अन्तर्भाव "fwht.hpp"

#अन्तर्भाव <cmath>
#define P 1.0f
#define N -1.0f

// constant Hadamard matrix via Paley I construction
स्थिर constexpr भिन्न H12[12][12] = {
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

स्थिर constexpr भिन्न H20[20][20] = {
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

template <पूर्णाङ्क N, typename T>
स्थिर शून्य fwht_kernel(ध्रुव T * __restrict__ src, भिन्न * __restrict__ dst, ध्रुव int64_t n_rows,
                        ध्रुव भिन्न scale, ध्रुव sycl::nd_item<2> & item) {
    ध्रुव sycl::sub_group sg = item.get_sub_group();

    ध्रुव int64_t r = item.get_global_id(0);
    जइ (r >= n_rows) {
        प्रत्यागम;
    }

    src += r * N;
    dst += r * N;

    constexpr पूर्णाङ्क el_w = N / WARP_SIZE;
    static_assert(el_w >= 1 && N % WARP_SIZE == 0, "row must be a whole number of sub-group widths");

    भिन्न     reg[el_w];
    ध्रुव पूर्णाङ्क lane = sg.get_local_linear_id();

#pragma unroll
    प्रति (पूर्णाङ्क i = 0; i < el_w; ++i) {
        reg[i] = static_cast<भिन्न>(src[i * WARP_SIZE + lane]) * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#pragma unroll
    प्रति (पूर्णाङ्क h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        प्रति (पूर्णाङ्क j = 0; j < el_w; ++j) {
            ध्रुव भिन्न val  = reg[j];
            ध्रुव भिन्न val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#pragma unroll
    प्रति (पूर्णाङ्क h = WARP_SIZE; h < N; h *= 2) {
        ध्रुव पूर्णाङ्क step = h / WARP_SIZE;
#pragma unroll
        प्रति (पूर्णाङ्क j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            प्रति (पूर्णाङ्क k = 0; k < step; ++k) {
                ध्रुव भिन्न x = reg[j + k];
                ध्रुव भिन्न y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    प्रति (पूर्णाङ्क i = 0; i < el_w; ++i) {
        dst[i * WARP_SIZE + lane] = reg[i];
    }
}

template <पूर्णाङ्क N, typename T>
स्थिर शून्य launch_fwht(ध्रुव T * src, भिन्न * dst, ध्रुव int64_t n_rows, ध्रुव भिन्न scale,
                        dpct::queue_ptr stream) {
    constexpr पूर्णाङ्क rows_per_block = 4;

    ध्रुव int64_t अङ्कः_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ध्रुव sycl::range<2> global(अङ्कः_blocks * rows_per_block, WARP_SIZE);
    ध्रुव sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel<N, T>(src, dst, n_rows, scale, item);
                         });
}

// Wide blocks: one row per work-group instead of per sub-group, so each work-item
// keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
// still shuffle; those up to NT go through work-group local memory; the rest stay
// in registers.
template <पूर्णाङ्क N, पूर्णाङ्क NT, typename T>
स्थिर शून्य fwht_kernel_wide(ध्रुव T * __restrict__ src,
                             भिन्न * __restrict__ dst,
                             ध्रुव int64_t            n_rows,
                             ध्रुव भिन्न              scale,
                             ध्रुव sycl::nd_item<2> & item,
                             भिन्न *                  smem) {
    ध्रुव int64_t r = item.get_global_id(0);
    जइ (r >= n_rows) {
        प्रत्यागम;
    }

    src += r * N;
    dst += r * N;

    constexpr पूर्णाङ्क el_w = N / NT;
    static_assert(el_w >= 1 && N % NT == 0, "row must be a whole number of work-group widths");

    ध्रुव पूर्णाङ्क tid = item.get_local_id(1);

    भिन्न reg[el_w];
#pragma unroll
    प्रति (पूर्णाङ्क i = 0; i < el_w; ++i) {
        reg[i] = static_cast<भिन्न>(src[i * NT + tid]) * scale;
    }

    ध्रुव sycl::sub_group sg   = item.get_sub_group();
    ध्रुव पूर्णाङ्क             lane = sg.get_local_linear_id();

    // Butterflies inside the sub-group, same pattern as the narrow kernel.
#pragma unroll
    प्रति (पूर्णाङ्क h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        प्रति (पूर्णाङ्क j = 0; j < el_w; ++j) {
            ध्रुव भिन्न val  = reg[j];
            ध्रुव भिन्न val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies from the sub-group width up to NT: the partner lane is outside
    // this sub-group, so it goes through work-group local memory instead of a shuffle.
    प्रति (पूर्णाङ्क h = WARP_SIZE; h < NT; h *= 2) {
#pragma unroll
        प्रति (पूर्णाङ्क j = 0; j < el_w; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        प्रति (पूर्णाङ्क j = 0; j < el_w; ++j) {
            ध्रुव भिन्न val  = reg[j];
            ध्रुव भिन्न val2 = smem[j * NT + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    // Butterflies across registers: h is a multiple of NT, so the partner of element
    // i*NT + tid lives in reg[i + h/NT] on the same work-item.
    प्रति (पूर्णाङ्क h = NT; h < N; h *= 2) {
        ध्रुव पूर्णाङ्क step = h / NT;
        प्रति (पूर्णाङ्क j = 0; j < el_w; j += 2 * step) {
            प्रति (पूर्णाङ्क k = 0; k < step; ++k) {
                ध्रुव भिन्न x = reg[j + k];
                ध्रुव भिन्न y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    प्रति (पूर्णाङ्क i = 0; i < el_w; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

template <पूर्णाङ्क N, पूर्णाङ्क NT, typename T>
स्थिर शून्य launch_fwht_wide(ध्रुव T *       src,
                             भिन्न *         dst,
                             ध्रुव int64_t   n_rows,
                             ध्रुव भिन्न     scale,
                             dpct::queue_ptr stream) {
    ध्रुव sycl::range<2> global(n_rows, NT);
    ध्रुव sycl::range<2> local(1, NT);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<भिन्न, 1> smem(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel_wide<N, NT, T>(src, dst, n_rows, scale, item, get_pointer(smem));
                         });
    });
}

template <पूर्णाङ्क N, पूर्णाङ्क m, typename T>
स्थिर शून्य kronecker_kernel(ध्रुव T * __restrict__ src,
                             भिन्न * __restrict__ dst,
                             ध्रुव int64_t            n_rows,
                             ध्रुव भिन्न              scale,
                             ध्रुव sycl::nd_item<2> & item) {
    static_assert(m == 12 || m == 20, "block size has to be 12 or 20.");

    ध्रुव sycl::sub_group sg = item.get_sub_group();

    ध्रुव int64_t r = item.get_global_id(0);
    जइ (r >= n_rows) {
        प्रत्यागम;
    }

    src += r * N;
    dst += r * N;

    constexpr पूर्णाङ्क blocks_per_group = N / m;
    constexpr पूर्णाङ्क el_w             = blocks_per_group / WARP_SIZE;
    static_assert(el_w >= 1 && blocks_per_group % WARP_SIZE == 0, "blocks_per_group must be a multiple of WARP_SIZE");
    भिन्न     reg[el_w * m];
    ध्रुव पूर्णाङ्क lane = sg.get_local_linear_id();

#pragma unroll
    प्रति (पूर्णाङ्क i = 0; i < el_w; ++i) {
        ध्रुव पूर्णाङ्क b_idx = i * WARP_SIZE + lane;

#pragma unroll
        प्रति (पूर्णाङ्क j = 0; j < m; ++j) {
            reg[i * m + j] = static_cast<भिन्न>(src[b_idx * m + j]) * scale;
        }
    }

#pragma unroll
    प्रति (पूर्णाङ्क b = 0; b < el_w; ++b) {
        भिन्न z[m] = { 0.0f };

#pragma unroll
        प्रति (पूर्णाङ्क i = 0; i < m; ++i) {
#pragma unroll
            प्रति (पूर्णाङ्क j = 0; j < m; ++j) {
                ध्रुव भिन्न h = (m == 12 ? H12[j][i] : H20[j][i]);
                z[i] += reg[b * m + j] * h;
            }
        }

#pragma unroll
        प्रति (पूर्णाङ्क i = 0; i < m; ++i) {
            reg[b * m + i] = z[i];
        }
    }

#pragma unroll
    प्रति (पूर्णाङ्क h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        प्रति (पूर्णाङ्क j = 0; j < el_w; ++j) {
#pragma unroll
            प्रति (पूर्णाङ्क k = 0; k < m; ++k) {
                ध्रुव भिन्न val  = reg[j * m + k];
                ध्रुव भिन्न val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

                reg[j * m + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#pragma unroll
    प्रति (पूर्णाङ्क h = WARP_SIZE; h < blocks_per_group; h *= 2) {
        ध्रुव पूर्णाङ्क step = h / WARP_SIZE;
#pragma unroll
        प्रति (पूर्णाङ्क j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            प्रति (पूर्णाङ्क s = 0; s < step; ++s) {
#pragma unroll
                प्रति (पूर्णाङ्क k = 0; k < m; ++k) {
                    ध्रुव भिन्न x = reg[(j + s) * m + k];
                    ध्रुव भिन्न y = reg[(j + s + step) * m + k];

                    reg[(j + s) * m + k]        = x + y;
                    reg[(j + s + step) * m + k] = x - y;
                }
            }
        }
    }

#pragma unroll
    प्रति (पूर्णाङ्क i = 0; i < el_w; ++i) {
        ध्रुव पूर्णाङ्क b_idx = i * WARP_SIZE + lane;
#pragma unroll
        प्रति (पूर्णाङ्क k = 0; k < m; ++k) {
            dst[b_idx * m + k] = reg[i * m + k];
        }
    }
}

template <पूर्णाङ्क N, पूर्णाङ्क m, typename T>
स्थिर शून्य launch_kronecker(ध्रुव T *       src,
                             भिन्न *         dst,
                             ध्रुव int64_t   n_rows,
                             ध्रुव भिन्न     scale,
                             dpct::queue_ptr stream) {
    constexpr पूर्णाङ्क rows_per_block = 4;

    ध्रुव int64_t अङ्कः_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ध्रुव sycl::range<2> global(अङ्कः_blocks * rows_per_block, WARP_SIZE);
    ध्रुव sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             kronecker_kernel<N, m, T>(src, dst, n_rows, scale, item);
                         });
}

template <typename T>
स्थिर तार्किक ggml_sycl_op_fwht_impl(ggml_backend_sycl_context & ctx, ध्रुव ggml_tensor * src, ggml_tensor * dst) {
    ध्रुव पूर्णाङ्क     n    = (पूर्णाङ्क) src->ne[0];
    ध्रुव int64_t rows = ggml_nrows(src);

    ध्रुव T *       src_d  = (ध्रुव T *) src->data;
    भिन्न *         dst_d  = (भिन्न *) dst->data;
    dpct::queue_ptr stream = ctx.stream();

    ध्रुव भिन्न scale = 1.0f / std::sqrt((भिन्न) n);

    विकल्प (n) {
        पक्ष 64:
            launch_fwht<64>(src_d, dst_d, rows, scale, stream);
            प्रत्यागम सच्च;
        पक्ष 128:
            launch_fwht<128>(src_d, dst_d, rows, scale, stream);
            प्रत्यागम सच्च;
        पक्ष 256:
            launch_fwht<256>(src_d, dst_d, rows, scale, stream);
            प्रत्यागम सच्च;
        पक्ष 512:
            launch_fwht<512>(src_d, dst_d, rows, scale, stream);
            प्रत्यागम सच्च;
        पक्ष 384:
            launch_kronecker<384, 12>(src_d, dst_d, rows, scale, stream);
            प्रत्यागम सच्च;
        पक्ष 768:
            launch_kronecker<768, 12>(src_d, dst_d, rows, scale, stream);
            प्रत्यागम सच्च;
        पक्ष 640:
            launch_kronecker<640, 20>(src_d, dst_d, rows, scale, stream);
            प्रत्यागम सच्च;
        पक्ष 1280:
            launch_kronecker<1280, 20>(src_d, dst_d, rows, scale, stream);
            प्रत्यागम सच्च;
        पक्ष 1024:
            launch_fwht_wide<1024, 256>(src_d, dst_d, rows, scale, stream);
            प्रत्यागम सच्च;
        पक्ष 2048:
            launch_fwht_wide<2048, 256>(src_d, dst_d, rows, scale, stream);
            प्रत्यागम सच्च;
        पक्ष 4096:
            launch_fwht_wide<4096, 256>(src_d, dst_d, rows, scale, stream);
            प्रत्यागम सच्च;
        पक्ष 8192:
            launch_fwht_wide<8192, 256>(src_d, dst_d, rows, scale, stream);
            प्रत्यागम सच्च;
        स्वभाव:
            प्रत्यागम मिच्छ;
    }
}

तार्किक ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, ध्रुव ggml_tensor * src, ggml_tensor * dst) {
    जइ (dst->type != GGML_TYPE_F32) {
        प्रत्यागम मिच्छ;
    }
    जइ (!ggml_are_same_shape(src, dst)) {
        प्रत्यागम मिच्छ;
    }
    जइ (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        प्रत्यागम मिच्छ;
    }

    विकल्प (src->type) {
        पक्ष GGML_TYPE_F32:
            प्रत्यागम ggml_sycl_op_fwht_impl<भिन्न>(ctx, src, dst);
        पक्ष GGML_TYPE_F16:
            प्रत्यागम ggml_sycl_op_fwht_impl<sycl::half>(ctx, src, dst);
        स्वभाव:
            प्रत्यागम मिच्छ;
    }
}
