#അന്തര്ഭാവ "fwht.hpp"

#അന്തര്ഭാവ <cmath>
#define P 1.0f
#define N -1.0f

// constant Hadamard matrix via Paley I construction
സ്ഥിതം constexpr ദശാംശം H12[12][12] = {
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

സ്ഥിതം constexpr ദശാംശം H20[20][20] = {
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

template <പൂർണ്ണസംഖ്യ N, typename T>
സ്ഥിതം ശൂന്യം fwht_kernel(സ്ഥിരം T * __restrict__ src, ദശാംശം * __restrict__ dst, സ്ഥിരം int64_t n_rows,
                        സ്ഥിരം ദശാംശം scale, സ്ഥിരം sycl::nd_item<2> & item) {
    സ്ഥിരം sycl::sub_group sg = item.get_sub_group();

    സ്ഥിരം int64_t r = item.get_global_id(0);
    എങ്കിൽ (r >= n_rows) {
        തിരികെ;
    }

    src += r * N;
    dst += r * N;

    constexpr പൂർണ്ണസംഖ്യ el_w = N / WARP_SIZE;
    static_assert(el_w >= 1 && N % WARP_SIZE == 0, "row must be a whole number of sub-group widths");

    ദശാംശം     reg[el_w];
    സ്ഥിരം പൂർണ്ണസംഖ്യ lane = sg.get_local_linear_id();

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < el_w; ++i) {
        reg[i] = static_cast<ദശാംശം>(src[i * WARP_SIZE + lane]) * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < el_w; ++j) {
            സ്ഥിരം ദശാംശം val  = reg[j];
            സ്ഥിരം ദശാംശം val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ h = WARP_SIZE; h < N; h *= 2) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ step = h / WARP_SIZE;
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < step; ++k) {
                സ്ഥിരം ദശാംശം x = reg[j + k];
                സ്ഥിരം ദശാംശം y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < el_w; ++i) {
        dst[i * WARP_SIZE + lane] = reg[i];
    }
}

template <പൂർണ്ണസംഖ്യ N, typename T>
സ്ഥിതം ശൂന്യം launch_fwht(സ്ഥിരം T * src, ദശാംശം * dst, സ്ഥിരം int64_t n_rows, സ്ഥിരം ദശാംശം scale,
                        dpct::queue_ptr stream) {
    constexpr പൂർണ്ണസംഖ്യ rows_per_block = 4;

    സ്ഥിരം int64_t അക്കം_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    സ്ഥിരം sycl::range<2> global(അക്കം_blocks * rows_per_block, WARP_SIZE);
    സ്ഥിരം sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel<N, T>(src, dst, n_rows, scale, item);
                         });
}

// Wide blocks: one row per work-group instead of per sub-group, so each work-item
// keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
// still shuffle; those up to NT go through work-group local memory; the rest stay
// in registers.
template <പൂർണ്ണസംഖ്യ N, പൂർണ്ണസംഖ്യ NT, typename T>
സ്ഥിതം ശൂന്യം fwht_kernel_wide(സ്ഥിരം T * __restrict__ src,
                             ദശാംശം * __restrict__ dst,
                             സ്ഥിരം int64_t            n_rows,
                             സ്ഥിരം ദശാംശം              scale,
                             സ്ഥിരം sycl::nd_item<2> & item,
                             ദശാംശം *                  smem) {
    സ്ഥിരം int64_t r = item.get_global_id(0);
    എങ്കിൽ (r >= n_rows) {
        തിരികെ;
    }

    src += r * N;
    dst += r * N;

    constexpr പൂർണ്ണസംഖ്യ el_w = N / NT;
    static_assert(el_w >= 1 && N % NT == 0, "row must be a whole number of work-group widths");

    സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item.get_local_id(1);

    ദശാംശം reg[el_w];
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < el_w; ++i) {
        reg[i] = static_cast<ദശാംശം>(src[i * NT + tid]) * scale;
    }

    സ്ഥിരം sycl::sub_group sg   = item.get_sub_group();
    സ്ഥിരം പൂർണ്ണസംഖ്യ             lane = sg.get_local_linear_id();

    // Butterflies inside the sub-group, same pattern as the narrow kernel.
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < el_w; ++j) {
            സ്ഥിരം ദശാംശം val  = reg[j];
            സ്ഥിരം ദശാംശം val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies from the sub-group width up to NT: the partner lane is outside
    // this sub-group, so it goes through work-group local memory instead of a shuffle.
    വേണ്ടി (പൂർണ്ണസംഖ്യ h = WARP_SIZE; h < NT; h *= 2) {
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < el_w; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < el_w; ++j) {
            സ്ഥിരം ദശാംശം val  = reg[j];
            സ്ഥിരം ദശാംശം val2 = smem[j * NT + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    // Butterflies across registers: h is a multiple of NT, so the partner of element
    // i*NT + tid lives in reg[i + h/NT] on the same work-item.
    വേണ്ടി (പൂർണ്ണസംഖ്യ h = NT; h < N; h *= 2) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ step = h / NT;
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < el_w; j += 2 * step) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < step; ++k) {
                സ്ഥിരം ദശാംശം x = reg[j + k];
                സ്ഥിരം ദശാംശം y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < el_w; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

template <പൂർണ്ണസംഖ്യ N, പൂർണ്ണസംഖ്യ NT, typename T>
സ്ഥിതം ശൂന്യം launch_fwht_wide(സ്ഥിരം T *       src,
                             ദശാംശം *         dst,
                             സ്ഥിരം int64_t   n_rows,
                             സ്ഥിരം ദശാംശം     scale,
                             dpct::queue_ptr stream) {
    സ്ഥിരം sycl::range<2> global(n_rows, NT);
    സ്ഥിരം sycl::range<2> local(1, NT);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<ദശാംശം, 1> smem(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel_wide<N, NT, T>(src, dst, n_rows, scale, item, get_pointer(smem));
                         });
    });
}

template <പൂർണ്ണസംഖ്യ N, പൂർണ്ണസംഖ്യ m, typename T>
സ്ഥിതം ശൂന്യം kronecker_kernel(സ്ഥിരം T * __restrict__ src,
                             ദശാംശം * __restrict__ dst,
                             സ്ഥിരം int64_t            n_rows,
                             സ്ഥിരം ദശാംശം              scale,
                             സ്ഥിരം sycl::nd_item<2> & item) {
    static_assert(m == 12 || m == 20, "block size has to be 12 or 20.");

    സ്ഥിരം sycl::sub_group sg = item.get_sub_group();

    സ്ഥിരം int64_t r = item.get_global_id(0);
    എങ്കിൽ (r >= n_rows) {
        തിരികെ;
    }

    src += r * N;
    dst += r * N;

    constexpr പൂർണ്ണസംഖ്യ blocks_per_group = N / m;
    constexpr പൂർണ്ണസംഖ്യ el_w             = blocks_per_group / WARP_SIZE;
    static_assert(el_w >= 1 && blocks_per_group % WARP_SIZE == 0, "blocks_per_group must be a multiple of WARP_SIZE");
    ദശാംശം     reg[el_w * m];
    സ്ഥിരം പൂർണ്ണസംഖ്യ lane = sg.get_local_linear_id();

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < el_w; ++i) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ b_idx = i * WARP_SIZE + lane;

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < m; ++j) {
            reg[i * m + j] = static_cast<ദശാംശം>(src[b_idx * m + j]) * scale;
        }
    }

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < el_w; ++b) {
        ദശാംശം z[m] = { 0.0f };

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < m; ++i) {
#pragma unroll
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < m; ++j) {
                സ്ഥിരം ദശാംശം h = (m == 12 ? H12[j][i] : H20[j][i]);
                z[i] += reg[b * m + j] * h;
            }
        }

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < m; ++i) {
            reg[b * m + i] = z[i];
        }
    }

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < el_w; ++j) {
#pragma unroll
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < m; ++k) {
                സ്ഥിരം ദശാംശം val  = reg[j * m + k];
                സ്ഥിരം ദശാംശം val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

                reg[j * m + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ h = WARP_SIZE; h < blocks_per_group; h *= 2) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ step = h / WARP_SIZE;
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            വേണ്ടി (പൂർണ്ണസംഖ്യ s = 0; s < step; ++s) {
#pragma unroll
                വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < m; ++k) {
                    സ്ഥിരം ദശാംശം x = reg[(j + s) * m + k];
                    സ്ഥിരം ദശാംശം y = reg[(j + s + step) * m + k];

                    reg[(j + s) * m + k]        = x + y;
                    reg[(j + s + step) * m + k] = x - y;
                }
            }
        }
    }

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < el_w; ++i) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ b_idx = i * WARP_SIZE + lane;
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < m; ++k) {
            dst[b_idx * m + k] = reg[i * m + k];
        }
    }
}

template <പൂർണ്ണസംഖ്യ N, പൂർണ്ണസംഖ്യ m, typename T>
സ്ഥിതം ശൂന്യം launch_kronecker(സ്ഥിരം T *       src,
                             ദശാംശം *         dst,
                             സ്ഥിരം int64_t   n_rows,
                             സ്ഥിരം ദശാംശം     scale,
                             dpct::queue_ptr stream) {
    constexpr പൂർണ്ണസംഖ്യ rows_per_block = 4;

    സ്ഥിരം int64_t അക്കം_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    സ്ഥിരം sycl::range<2> global(അക്കം_blocks * rows_per_block, WARP_SIZE);
    സ്ഥിരം sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             kronecker_kernel<N, m, T>(src, dst, n_rows, scale, item);
                         });
}

template <typename T>
സ്ഥിതം തർക്കം ggml_sycl_op_fwht_impl(ggml_backend_sycl_context & ctx, സ്ഥിരം ggml_tensor * src, ggml_tensor * dst) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ     n    = (പൂർണ്ണസംഖ്യ) src->ne[0];
    സ്ഥിരം int64_t rows = ggml_nrows(src);

    സ്ഥിരം T *       src_d  = (സ്ഥിരം T *) src->data;
    ദശാംശം *         dst_d  = (ദശാംശം *) dst->data;
    dpct::queue_ptr stream = ctx.stream();

    സ്ഥിരം ദശാംശം scale = 1.0f / std::sqrt((ദശാംശം) n);

    തിരഞ്ഞെടുപ്പ് (n) {
        അവസ്ഥ 64:
            launch_fwht<64>(src_d, dst_d, rows, scale, stream);
            തിരികെ സത്യം;
        അവസ്ഥ 128:
            launch_fwht<128>(src_d, dst_d, rows, scale, stream);
            തിരികെ സത്യം;
        അവസ്ഥ 256:
            launch_fwht<256>(src_d, dst_d, rows, scale, stream);
            തിരികെ സത്യം;
        അവസ്ഥ 512:
            launch_fwht<512>(src_d, dst_d, rows, scale, stream);
            തിരികെ സത്യം;
        അവസ്ഥ 384:
            launch_kronecker<384, 12>(src_d, dst_d, rows, scale, stream);
            തിരികെ സത്യം;
        അവസ്ഥ 768:
            launch_kronecker<768, 12>(src_d, dst_d, rows, scale, stream);
            തിരികെ സത്യം;
        അവസ്ഥ 640:
            launch_kronecker<640, 20>(src_d, dst_d, rows, scale, stream);
            തിരികെ സത്യം;
        അവസ്ഥ 1280:
            launch_kronecker<1280, 20>(src_d, dst_d, rows, scale, stream);
            തിരികെ സത്യം;
        അവസ്ഥ 1024:
            launch_fwht_wide<1024, 256>(src_d, dst_d, rows, scale, stream);
            തിരികെ സത്യം;
        അവസ്ഥ 2048:
            launch_fwht_wide<2048, 256>(src_d, dst_d, rows, scale, stream);
            തിരികെ സത്യം;
        അവസ്ഥ 4096:
            launch_fwht_wide<4096, 256>(src_d, dst_d, rows, scale, stream);
            തിരികെ സത്യം;
        അവസ്ഥ 8192:
            launch_fwht_wide<8192, 256>(src_d, dst_d, rows, scale, stream);
            തിരികെ സത്യം;
        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം;
    }
}

തർക്കം ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, സ്ഥിരം ggml_tensor * src, ggml_tensor * dst) {
    എങ്കിൽ (dst->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (!ggml_are_same_shape(src, dst)) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        തിരികെ അസത്യം;
    }

    തിരഞ്ഞെടുപ്പ് (src->type) {
        അവസ്ഥ GGML_TYPE_F32:
            തിരികെ ggml_sycl_op_fwht_impl<ദശാംശം>(ctx, src, dst);
        അവസ്ഥ GGML_TYPE_F16:
            തിരികെ ggml_sycl_op_fwht_impl<sycl::half>(ctx, src, dst);
        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം;
    }
}
