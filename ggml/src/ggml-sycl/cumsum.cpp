#અન્તર્ભાવ "cumsum.hpp"
#અન્તર્ભાવ "common.hpp"

#અન્તર્ભાવ <algorithm>

#define SYCL_CUMSUM_BLOCK_SIZE 256

સ્થિર __dpct_inline__ દશાંશ warp_prefix_inclusive_sum_f32(દશાંશ x, અચલ sycl::nd_item<3> & item) {
    પાછા sycl::inclusive_scan_over_group(item.get_sub_group(), x, sycl::plus<દશાંશ>());
}

સ્થિર શૂન્ય cumsum_f32_kernel(
        અચલ દશાંશ * __restrict__ src, દશાંશ * __restrict__ dst,
        અચલ int64_t ne00, અચલ int64_t ne01, અચલ int64_t ne02, અચલ int64_t ne03,
        અચલ int64_t s01, અચલ int64_t s02, અચલ int64_t s03,
        અચલ int64_t  d1, અચલ int64_t  d2, અચલ int64_t  d3,
        અચલ sycl::nd_item<3> & item, દશાંશ * smem) {

    અચલ પૂર્ણાંક tid = item.get_local_id(2);
    અચલ પૂર્ણાંક block_size = item.get_local_range(2);
    અચલ પૂર્ણાંક lane = tid % WARP_SIZE;
    અચલ પૂર્ણાંક warp = tid / WARP_SIZE;
    અચલ પૂર્ણાંક warps_per_block = block_size / WARP_SIZE;

    દશાંશ * s_vals      = smem;
    દશાંશ * s_warp_sums = smem + block_size;
    દશાંશ * s_carry     = smem + block_size + warps_per_block;

    જો (tid == 0) {
        s_carry[0] = 0.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    અચલ int64_t i3 = item.get_group(0);
    અચલ int64_t i2 = item.get_group(1);
    અચલ int64_t i1 = item.get_group(2);
    જો (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        પાછા;
    }

    અચલ દશાંશ * src_row = src + i1 * s01 + i2 * s02 + i3 * s03;
    દશાંશ       * dst_row = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr પૂર્ણાંક અંક_unroll = 4;
    દશાંશ temp[અંક_unroll];

    માટે (int64_t i = 0; i < ne00; i += અંક_unroll * block_size) {
        int64_t idx = i + tid * અંક_unroll;

        temp[0] = (idx < ne00 ? src_row[idx] : 0.0f);
#pragma unroll
        માટે (પૂર્ણાંક j = 1; j < અંક_unroll; j++) {
            temp[j] = temp[j - 1];
            જો (idx + j < ne00) {
                temp[j] += src_row[idx + j];
            }
        }

        દશાંશ val = (idx < ne00) ? temp[અંક_unroll - 1] : 0.0f;

        val = warp_prefix_inclusive_sum_f32(val, item);
        s_vals[tid] = val;

        જો (lane == WARP_SIZE - 1) {
            s_warp_sums[warp] = val;
        }
        item.barrier(sycl::access::fence_space::local_space);

        જો (warp == 0) {
            દશાંશ w = (tid < warps_per_block) ? s_warp_sums[tid] : 0.0f;
            દશાંશ inc = warp_prefix_inclusive_sum_f32(w, item);
            જો (tid < warps_per_block) {
                s_warp_sums[tid] = inc - w;
            }
            જો (tid == warps_per_block - 1) {
                s_carry[1] = inc;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        દશાંશ carry = s_carry[0];
        દશાંશ final_offset = s_vals[tid] + s_warp_sums[warp] + carry - temp[અંક_unroll - 1];

#pragma unroll
        માટે (પૂર્ણાંક j = 0; j < અંક_unroll; j++) {
            જો (idx + j < ne00) {
                dst_row[idx + j] = temp[j] + final_offset;
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        જો (tid == 0) {
            s_carry[0] += s_carry[1];
        }
    }
}

અંતઃરેખ શૂન્ય ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    અચલ દશાંશ * src_d = static_cast<અચલ દશાંશ *>(src0->data);
    દશાંશ       * dst_d = static_cast<દશાંશ *>(dst->data);

    અચલ int64_t ne00 = src0->ne[0];
    અચલ int64_t ne01 = src0->ne[1];
    અચલ int64_t ne02 = src0->ne[2];
    અચલ int64_t ne03 = src0->ne[3];

    અચલ size_t ts = કદ(દશાંશ);
    અચલ int64_t s01 = src0->nb[1] / ts;
    અચલ int64_t s02 = src0->nb[2] / ts;
    અચલ int64_t s03 = src0->nb[3] / ts;
    અચલ int64_t d1  = dst->nb[1] / ts;
    અચલ int64_t d2  = dst->nb[2] / ts;
    અચલ int64_t d3  = dst->nb[3] / ts;

    અચલ પૂર્ણાંક અંક_warps = (ne00 + WARP_SIZE - 1) / WARP_SIZE;
    પૂર્ણાંક block_size = અંક_warps * WARP_SIZE;
    block_size = std::min(block_size, SYCL_CUMSUM_BLOCK_SIZE);
    અચલ પૂર્ણાંક warps_per_block = block_size / WARP_SIZE;
    અચલ પૂર્ણાંક smem_size = block_size + warps_per_block + 2;

    અચલ sycl::range<3> grid(ne03, ne02, ne01);
    અચલ sycl::range<3> block(1, 1, block_size);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<દશાંશ, 1> smem_acc(sycl::range<1>(smem_size), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  item, get_pointer(smem_acc));
            });
    });
}

શૂન્ય ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cumsum(ctx, dst);
}
