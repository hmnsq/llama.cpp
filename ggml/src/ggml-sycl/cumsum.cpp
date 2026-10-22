#അന്തര്ഭാവ "cumsum.hpp"
#അന്തര്ഭാവ "common.hpp"

#അന്തര്ഭാവ <algorithm>

#define SYCL_CUMSUM_BLOCK_SIZE 256

സ്ഥിതം __dpct_inline__ ദശാംശം warp_prefix_inclusive_sum_f32(ദശാംശം x, സ്ഥിരം sycl::nd_item<3> & item) {
    തിരികെ sycl::inclusive_scan_over_group(item.get_sub_group(), x, sycl::plus<ദശാംശം>());
}

സ്ഥിതം ശൂന്യം cumsum_f32_kernel(
        സ്ഥിരം ദശാംശം * __restrict__ src, ദശാംശം * __restrict__ dst,
        സ്ഥിരം int64_t ne00, സ്ഥിരം int64_t ne01, സ്ഥിരം int64_t ne02, സ്ഥിരം int64_t ne03,
        സ്ഥിരം int64_t s01, സ്ഥിരം int64_t s02, സ്ഥിരം int64_t s03,
        സ്ഥിരം int64_t  d1, സ്ഥിരം int64_t  d2, സ്ഥിരം int64_t  d3,
        സ്ഥിരം sycl::nd_item<3> & item, ദശാംശം * smem) {

    സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item.get_local_id(2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_size = item.get_local_range(2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ lane = tid % WARP_SIZE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ warp = tid / WARP_SIZE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ warps_per_block = block_size / WARP_SIZE;

    ദശാംശം * s_vals      = smem;
    ദശാംശം * s_warp_sums = smem + block_size;
    ദശാംശം * s_carry     = smem + block_size + warps_per_block;

    എങ്കിൽ (tid == 0) {
        s_carry[0] = 0.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    സ്ഥിരം int64_t i3 = item.get_group(0);
    സ്ഥിരം int64_t i2 = item.get_group(1);
    സ്ഥിരം int64_t i1 = item.get_group(2);
    എങ്കിൽ (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        തിരികെ;
    }

    സ്ഥിരം ദശാംശം * src_row = src + i1 * s01 + i2 * s02 + i3 * s03;
    ദശാംശം       * dst_row = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr പൂർണ്ണസംഖ്യ അക്കം_unroll = 4;
    ദശാംശം temp[അക്കം_unroll];

    വേണ്ടി (int64_t i = 0; i < ne00; i += അക്കം_unroll * block_size) {
        int64_t idx = i + tid * അക്കം_unroll;

        temp[0] = (idx < ne00 ? src_row[idx] : 0.0f);
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 1; j < അക്കം_unroll; j++) {
            temp[j] = temp[j - 1];
            എങ്കിൽ (idx + j < ne00) {
                temp[j] += src_row[idx + j];
            }
        }

        ദശാംശം val = (idx < ne00) ? temp[അക്കം_unroll - 1] : 0.0f;

        val = warp_prefix_inclusive_sum_f32(val, item);
        s_vals[tid] = val;

        എങ്കിൽ (lane == WARP_SIZE - 1) {
            s_warp_sums[warp] = val;
        }
        item.barrier(sycl::access::fence_space::local_space);

        എങ്കിൽ (warp == 0) {
            ദശാംശം w = (tid < warps_per_block) ? s_warp_sums[tid] : 0.0f;
            ദശാംശം inc = warp_prefix_inclusive_sum_f32(w, item);
            എങ്കിൽ (tid < warps_per_block) {
                s_warp_sums[tid] = inc - w;
            }
            എങ്കിൽ (tid == warps_per_block - 1) {
                s_carry[1] = inc;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        ദശാംശം carry = s_carry[0];
        ദശാംശം final_offset = s_vals[tid] + s_warp_sums[warp] + carry - temp[അക്കം_unroll - 1];

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < അക്കം_unroll; j++) {
            എങ്കിൽ (idx + j < ne00) {
                dst_row[idx + j] = temp[j] + final_offset;
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        എങ്കിൽ (tid == 0) {
            s_carry[0] += s_carry[1];
        }
    }
}

വരിയിൽ ശൂന്യം ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    സ്ഥിരം ദശാംശം * src_d = static_cast<സ്ഥിരം ദശാംശം *>(src0->data);
    ദശാംശം       * dst_d = static_cast<ദശാംശം *>(dst->data);

    സ്ഥിരം int64_t ne00 = src0->ne[0];
    സ്ഥിരം int64_t ne01 = src0->ne[1];
    സ്ഥിരം int64_t ne02 = src0->ne[2];
    സ്ഥിരം int64_t ne03 = src0->ne[3];

    സ്ഥിരം size_t ts = വലുപ്പം(ദശാംശം);
    സ്ഥിരം int64_t s01 = src0->nb[1] / ts;
    സ്ഥിരം int64_t s02 = src0->nb[2] / ts;
    സ്ഥിരം int64_t s03 = src0->nb[3] / ts;
    സ്ഥിരം int64_t d1  = dst->nb[1] / ts;
    സ്ഥിരം int64_t d2  = dst->nb[2] / ts;
    സ്ഥിരം int64_t d3  = dst->nb[3] / ts;

    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_warps = (ne00 + WARP_SIZE - 1) / WARP_SIZE;
    പൂർണ്ണസംഖ്യ block_size = അക്കം_warps * WARP_SIZE;
    block_size = std::min(block_size, SYCL_CUMSUM_BLOCK_SIZE);
    സ്ഥിരം പൂർണ്ണസംഖ്യ warps_per_block = block_size / WARP_SIZE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ smem_size = block_size + warps_per_block + 2;

    സ്ഥിരം sycl::range<3> grid(ne03, ne02, ne01);
    സ്ഥിരം sycl::range<3> block(1, 1, block_size);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<ദശാംശം, 1> smem_acc(sycl::range<1>(smem_size), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  item, get_pointer(smem_acc));
            });
    });
}

ശൂന്യം ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cumsum(ctx, dst);
}
