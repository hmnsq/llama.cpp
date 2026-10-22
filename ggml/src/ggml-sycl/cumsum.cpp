#अन्तर्भाव "cumsum.hpp"
#अन्तर्भाव "common.hpp"

#अन्तर्भाव <algorithm>

#define SYCL_CUMSUM_BLOCK_SIZE 256

स्थिर __dpct_inline__ भिन्न warp_prefix_inclusive_sum_f32(भिन्न x, ध्रुव sycl::nd_item<3> & item) {
    प्रत्यागम sycl::inclusive_scan_over_group(item.get_sub_group(), x, sycl::plus<भिन्न>());
}

स्थिर शून्य cumsum_f32_kernel(
        ध्रुव भिन्न * __restrict__ src, भिन्न * __restrict__ dst,
        ध्रुव int64_t ne00, ध्रुव int64_t ne01, ध्रुव int64_t ne02, ध्रुव int64_t ne03,
        ध्रुव int64_t s01, ध्रुव int64_t s02, ध्रुव int64_t s03,
        ध्रुव int64_t  d1, ध्रुव int64_t  d2, ध्रुव int64_t  d3,
        ध्रुव sycl::nd_item<3> & item, भिन्न * smem) {

    ध्रुव पूर्णाङ्क tid = item.get_local_id(2);
    ध्रुव पूर्णाङ्क block_size = item.get_local_range(2);
    ध्रुव पूर्णाङ्क lane = tid % WARP_SIZE;
    ध्रुव पूर्णाङ्क warp = tid / WARP_SIZE;
    ध्रुव पूर्णाङ्क warps_per_block = block_size / WARP_SIZE;

    भिन्न * s_vals      = smem;
    भिन्न * s_warp_sums = smem + block_size;
    भिन्न * s_carry     = smem + block_size + warps_per_block;

    यदि (tid == 0) {
        s_carry[0] = 0.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    ध्रुव int64_t i3 = item.get_group(0);
    ध्रुव int64_t i2 = item.get_group(1);
    ध्रुव int64_t i1 = item.get_group(2);
    यदि (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        प्रत्यागम;
    }

    ध्रुव भिन्न * src_row = src + i1 * s01 + i2 * s02 + i3 * s03;
    भिन्न       * dst_row = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr पूर्णाङ्क अंक_unroll = 4;
    भिन्न temp[अंक_unroll];

    प्रति (int64_t i = 0; i < ne00; i += अंक_unroll * block_size) {
        int64_t idx = i + tid * अंक_unroll;

        temp[0] = (idx < ne00 ? src_row[idx] : 0.0f);
#pragma unroll
        प्रति (पूर्णाङ्क j = 1; j < अंक_unroll; j++) {
            temp[j] = temp[j - 1];
            यदि (idx + j < ne00) {
                temp[j] += src_row[idx + j];
            }
        }

        भिन्न val = (idx < ne00) ? temp[अंक_unroll - 1] : 0.0f;

        val = warp_prefix_inclusive_sum_f32(val, item);
        s_vals[tid] = val;

        यदि (lane == WARP_SIZE - 1) {
            s_warp_sums[warp] = val;
        }
        item.barrier(sycl::access::fence_space::local_space);

        यदि (warp == 0) {
            भिन्न w = (tid < warps_per_block) ? s_warp_sums[tid] : 0.0f;
            भिन्न inc = warp_prefix_inclusive_sum_f32(w, item);
            यदि (tid < warps_per_block) {
                s_warp_sums[tid] = inc - w;
            }
            यदि (tid == warps_per_block - 1) {
                s_carry[1] = inc;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        भिन्न carry = s_carry[0];
        भिन्न final_offset = s_vals[tid] + s_warp_sums[warp] + carry - temp[अंक_unroll - 1];

#pragma unroll
        प्रति (पूर्णाङ्क j = 0; j < अंक_unroll; j++) {
            यदि (idx + j < ne00) {
                dst_row[idx + j] = temp[j] + final_offset;
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        यदि (tid == 0) {
            s_carry[0] += s_carry[1];
        }
    }
}

अन्तःस्थ शून्य ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ध्रुव भिन्न * src_d = static_cast<ध्रुव भिन्न *>(src0->data);
    भिन्न       * dst_d = static_cast<भिन्न *>(dst->data);

    ध्रुव int64_t ne00 = src0->ne[0];
    ध्रुव int64_t ne01 = src0->ne[1];
    ध्रुव int64_t ne02 = src0->ne[2];
    ध्रुव int64_t ne03 = src0->ne[3];

    ध्रुव size_t ts = परिमाण(भिन्न);
    ध्रुव int64_t s01 = src0->nb[1] / ts;
    ध्रुव int64_t s02 = src0->nb[2] / ts;
    ध्रुव int64_t s03 = src0->nb[3] / ts;
    ध्रुव int64_t d1  = dst->nb[1] / ts;
    ध्रुव int64_t d2  = dst->nb[2] / ts;
    ध्रुव int64_t d3  = dst->nb[3] / ts;

    ध्रुव पूर्णाङ्क अंक_warps = (ne00 + WARP_SIZE - 1) / WARP_SIZE;
    पूर्णाङ्क block_size = अंक_warps * WARP_SIZE;
    block_size = std::min(block_size, SYCL_CUMSUM_BLOCK_SIZE);
    ध्रुव पूर्णाङ्क warps_per_block = block_size / WARP_SIZE;
    ध्रुव पूर्णाङ्क smem_size = block_size + warps_per_block + 2;

    ध्रुव sycl::range<3> grid(ne03, ne02, ne01);
    ध्रुव sycl::range<3> block(1, 1, block_size);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<भिन्न, 1> smem_acc(sycl::range<1>(smem_size), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  item, get_pointer(smem_acc));
            });
    });
}

शून्य ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cumsum(ctx, dst);
}
