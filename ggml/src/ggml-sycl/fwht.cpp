#समावेश "fwht.hpp"

#समावेश <cmath>
#घोषणा P 1.0f
#घोषणा N -1.0f

// constant Hadamard matrix via Paley I construction
अटल constexpr भग्न H12[12][12] = {
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

अटल constexpr भग्न H20[20][20] = {
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

#अघोषित P
#अघोषित N

ढाँचा <पूर्णांक N, typename T>
अटल व्योम fwht_kernel(स्थिर T * __restrict__ src, भग्न * __restrict__ dst, स्थिर int64_t n_rows,
                        स्थिर भग्न scale, स्थिर sycl::nd_item<2> & item) {
    स्थिर sycl::sub_group sg = item.get_sub_group();

    स्थिर int64_t r = item.get_global_id(0);
    अगर (r >= n_rows) {
        वापस;
    }

    src += r * N;
    dst += r * N;

    constexpr पूर्णांक el_w = N / WARP_आकार;
    static_assert(el_w >= 1 && N % WARP_आकार == 0, "row must be a whole number of sub-group widths");

    भग्न     reg[el_w];
    स्थिर पूर्णांक lane = sg.get_local_linear_id();

#आशय unroll
    क्रम (पूर्णांक i = 0; i < el_w; ++i) {
        reg[i] = static_cast<भग्न>(src[i * WARP_आकार + lane]) * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#आशय unroll
    क्रम (पूर्णांक h = 1; h < WARP_आकार; h *= 2) {
#आशय unroll
        क्रम (पूर्णांक j = 0; j < el_w; ++j) {
            स्थिर भग्न val  = reg[j];
            स्थिर भग्न val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_आकार);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#आशय unroll
    क्रम (पूर्णांक h = WARP_आकार; h < N; h *= 2) {
        स्थिर पूर्णांक step = h / WARP_आकार;
#आशय unroll
        क्रम (पूर्णांक j = 0; j < el_w; j += 2 * step) {
#आशय unroll
            क्रम (पूर्णांक k = 0; k < step; ++k) {
                स्थिर भग्न x = reg[j + k];
                स्थिर भग्न y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#आशय unroll
    क्रम (पूर्णांक i = 0; i < el_w; ++i) {
        dst[i * WARP_आकार + lane] = reg[i];
    }
}

ढाँचा <पूर्णांक N, typename T>
अटल व्योम launch_fwht(स्थिर T * src, भग्न * dst, स्थिर int64_t n_rows, स्थिर भग्न scale,
                        dpct::queue_ptr stream) {
    constexpr पूर्णांक rows_per_block = 4;

    स्थिर int64_t अंक_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    स्थिर sycl::range<2> global(अंक_blocks * rows_per_block, WARP_आकार);
    स्थिर sycl::range<2> local(rows_per_block, WARP_आकार);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                             fwht_kernel<N, T>(src, dst, n_rows, scale, item);
                         });
}

// Wide blocks: one row per work-group instead of per sub-group, so each work-item
// keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
// still shuffle; those up to NT go through work-group local memory; the rest stay
// in registers.
ढाँचा <पूर्णांक N, पूर्णांक NT, typename T>
अटल व्योम fwht_kernel_wide(स्थिर T * __restrict__ src,
                             भग्न * __restrict__ dst,
                             स्थिर int64_t            n_rows,
                             स्थिर भग्न              scale,
                             स्थिर sycl::nd_item<2> & item,
                             भग्न *                  smem) {
    स्थिर int64_t r = item.get_global_id(0);
    अगर (r >= n_rows) {
        वापस;
    }

    src += r * N;
    dst += r * N;

    constexpr पूर्णांक el_w = N / NT;
    static_assert(el_w >= 1 && N % NT == 0, "row must be a whole number of work-group widths");

    स्थिर पूर्णांक tid = item.get_local_id(1);

    भग्न reg[el_w];
#आशय unroll
    क्रम (पूर्णांक i = 0; i < el_w; ++i) {
        reg[i] = static_cast<भग्न>(src[i * NT + tid]) * scale;
    }

    स्थिर sycl::sub_group sg   = item.get_sub_group();
    स्थिर पूर्णांक             lane = sg.get_local_linear_id();

    // Butterflies inside the sub-group, same pattern as the narrow kernel.
#आशय unroll
    क्रम (पूर्णांक h = 1; h < WARP_आकार; h *= 2) {
#आशय unroll
        क्रम (पूर्णांक j = 0; j < el_w; ++j) {
            स्थिर भग्न val  = reg[j];
            स्थिर भग्न val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_आकार);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies from the sub-group width up to NT: the partner lane is outside
    // this sub-group, so it goes through work-group local memory instead of a shuffle.
    क्रम (पूर्णांक h = WARP_आकार; h < NT; h *= 2) {
#आशय unroll
        क्रम (पूर्णांक j = 0; j < el_w; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        item.barrier(sycl::access::fence_space::local_space);
#आशय unroll
        क्रम (पूर्णांक j = 0; j < el_w; ++j) {
            स्थिर भग्न val  = reg[j];
            स्थिर भग्न val2 = smem[j * NT + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    // Butterflies across registers: h is a multiple of NT, so the partner of element
    // i*NT + tid lives in reg[i + h/NT] on the same work-item.
    क्रम (पूर्णांक h = NT; h < N; h *= 2) {
        स्थिर पूर्णांक step = h / NT;
        क्रम (पूर्णांक j = 0; j < el_w; j += 2 * step) {
            क्रम (पूर्णांक k = 0; k < step; ++k) {
                स्थिर भग्न x = reg[j + k];
                स्थिर भग्न y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#आशय unroll
    क्रम (पूर्णांक i = 0; i < el_w; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

ढाँचा <पूर्णांक N, पूर्णांक NT, typename T>
अटल व्योम launch_fwht_wide(स्थिर T *       src,
                             भग्न *         dst,
                             स्थिर int64_t   n_rows,
                             स्थिर भग्न     scale,
                             dpct::queue_ptr stream) {
    स्थिर sycl::range<2> global(n_rows, NT);
    स्थिर sycl::range<2> local(1, NT);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<भग्न, 1> smem(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                             fwht_kernel_wide<N, NT, T>(src, dst, n_rows, scale, item, get_pointer(smem));
                         });
    });
}

ढाँचा <पूर्णांक N, पूर्णांक m, typename T>
अटल व्योम kronecker_kernel(स्थिर T * __restrict__ src,
                             भग्न * __restrict__ dst,
                             स्थिर int64_t            n_rows,
                             स्थिर भग्न              scale,
                             स्थिर sycl::nd_item<2> & item) {
    static_assert(m == 12 || m == 20, "block size has to be 12 or 20.");

    स्थिर sycl::sub_group sg = item.get_sub_group();

    स्थिर int64_t r = item.get_global_id(0);
    अगर (r >= n_rows) {
        वापस;
    }

    src += r * N;
    dst += r * N;

    constexpr पूर्णांक blocks_per_group = N / m;
    constexpr पूर्णांक el_w             = blocks_per_group / WARP_आकार;
    static_assert(el_w >= 1 && blocks_per_group % WARP_आकार == 0, "blocks_per_group must be a multiple of WARP_SIZE");
    भग्न     reg[el_w * m];
    स्थिर पूर्णांक lane = sg.get_local_linear_id();

#आशय unroll
    क्रम (पूर्णांक i = 0; i < el_w; ++i) {
        स्थिर पूर्णांक b_idx = i * WARP_आकार + lane;

#आशय unroll
        क्रम (पूर्णांक j = 0; j < m; ++j) {
            reg[i * m + j] = static_cast<भग्न>(src[b_idx * m + j]) * scale;
        }
    }

#आशय unroll
    क्रम (पूर्णांक b = 0; b < el_w; ++b) {
        भग्न z[m] = { 0.0f };

#आशय unroll
        क्रम (पूर्णांक i = 0; i < m; ++i) {
#आशय unroll
            क्रम (पूर्णांक j = 0; j < m; ++j) {
                स्थिर भग्न h = (m == 12 ? H12[j][i] : H20[j][i]);
                z[i] += reg[b * m + j] * h;
            }
        }

#आशय unroll
        क्रम (पूर्णांक i = 0; i < m; ++i) {
            reg[b * m + i] = z[i];
        }
    }

#आशय unroll
    क्रम (पूर्णांक h = 1; h < WARP_आकार; h *= 2) {
#आशय unroll
        क्रम (पूर्णांक j = 0; j < el_w; ++j) {
#आशय unroll
            क्रम (पूर्णांक k = 0; k < m; ++k) {
                स्थिर भग्न val  = reg[j * m + k];
                स्थिर भग्न val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_आकार);

                reg[j * m + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#आशय unroll
    क्रम (पूर्णांक h = WARP_आकार; h < blocks_per_group; h *= 2) {
        स्थिर पूर्णांक step = h / WARP_आकार;
#आशय unroll
        क्रम (पूर्णांक j = 0; j < el_w; j += 2 * step) {
#आशय unroll
            क्रम (पूर्णांक s = 0; s < step; ++s) {
#आशय unroll
                क्रम (पूर्णांक k = 0; k < m; ++k) {
                    स्थिर भग्न x = reg[(j + s) * m + k];
                    स्थिर भग्न y = reg[(j + s + step) * m + k];

                    reg[(j + s) * m + k]        = x + y;
                    reg[(j + s + step) * m + k] = x - y;
                }
            }
        }
    }

#आशय unroll
    क्रम (पूर्णांक i = 0; i < el_w; ++i) {
        स्थिर पूर्णांक b_idx = i * WARP_आकार + lane;
#आशय unroll
        क्रम (पूर्णांक k = 0; k < m; ++k) {
            dst[b_idx * m + k] = reg[i * m + k];
        }
    }
}

ढाँचा <पूर्णांक N, पूर्णांक m, typename T>
अटल व्योम launch_kronecker(स्थिर T *       src,
                             भग्न *         dst,
                             स्थिर int64_t   n_rows,
                             स्थिर भग्न     scale,
                             dpct::queue_ptr stream) {
    constexpr पूर्णांक rows_per_block = 4;

    स्थिर int64_t अंक_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    स्थिर sycl::range<2> global(अंक_blocks * rows_per_block, WARP_आकार);
    स्थिर sycl::range<2> local(rows_per_block, WARP_आकार);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                             kronecker_kernel<N, m, T>(src, dst, n_rows, scale, item);
                         });
}

ढाँचा <typename T>
अटल bool ggml_sycl_op_fwht_impl(ggml_backend_sycl_context & ctx, स्थिर ggml_tensor * src, ggml_tensor * dst) {
    स्थिर पूर्णांक     n    = (पूर्णांक) src->ne[0];
    स्थिर int64_t rows = ggml_nrows(src);

    स्थिर T *       src_d  = (स्थिर T *) src->तथ्य;
    भग्न *         dst_d  = (भग्न *) dst->तथ्य;
    dpct::queue_ptr stream = ctx.stream();

    स्थिर भग्न scale = 1.0f / std::वर्ग_मूल((भग्न) n);

    चयन (n) {
        हाल 64:
            launch_fwht<64>(src_d, dst_d, rows, scale, stream);
            वापस true;
        हाल 128:
            launch_fwht<128>(src_d, dst_d, rows, scale, stream);
            वापस true;
        हाल 256:
            launch_fwht<256>(src_d, dst_d, rows, scale, stream);
            वापस true;
        हाल 512:
            launch_fwht<512>(src_d, dst_d, rows, scale, stream);
            वापस true;
        हाल 384:
            launch_kronecker<384, 12>(src_d, dst_d, rows, scale, stream);
            वापस true;
        हाल 768:
            launch_kronecker<768, 12>(src_d, dst_d, rows, scale, stream);
            वापस true;
        हाल 640:
            launch_kronecker<640, 20>(src_d, dst_d, rows, scale, stream);
            वापस true;
        हाल 1280:
            launch_kronecker<1280, 20>(src_d, dst_d, rows, scale, stream);
            वापस true;
        हाल 1024:
            launch_fwht_wide<1024, 256>(src_d, dst_d, rows, scale, stream);
            वापस true;
        हाल 2048:
            launch_fwht_wide<2048, 256>(src_d, dst_d, rows, scale, stream);
            वापस true;
        हाल 4096:
            launch_fwht_wide<4096, 256>(src_d, dst_d, rows, scale, stream);
            वापस true;
        हाल 8192:
            launch_fwht_wide<8192, 256>(src_d, dst_d, rows, scale, stream);
            वापस true;
        शेष:
            वापस false;
    }
}

bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, स्थिर ggml_tensor * src, ggml_tensor * dst) {
    अगर (dst->type != GGML_TYPE_F32) {
        वापस false;
    }
    अगर (!ggml_are_same_shape(src, dst)) {
        वापस false;
    }
    अगर (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        वापस false;
    }

    चयन (src->type) {
        हाल GGML_TYPE_F32:
            वापस ggml_sycl_op_fwht_impl<भग्न>(ctx, src, dst);
        हाल GGML_TYPE_F16:
            वापस ggml_sycl_op_fwht_impl<sycl::half>(ctx, src, dst);
        शेष:
            वापस false;
    }
}
