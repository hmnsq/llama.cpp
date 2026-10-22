#অন্তর্ভাৱ "cumsum.hpp"
#অন্তর্ভাৱ "common.hpp"

#অন্তর্ভাৱ <algorithm>

#define SYCL_CUMSUM_BLOCK_SIZE 256

স্থির __dpct_inline__ ভিন্ন warp_prefix_inclusive_sum_f32(ভিন্ন x, ধ্রুৱ sycl::nd_item<3> & item) {
    ঘূৰাই sycl::inclusive_scan_over_group(item.get_sub_group(), x, sycl::plus<ভিন্ন>());
}

স্থির শূন্য cumsum_f32_kernel(
        ধ্রুৱ ভিন্ন * __restrict__ src, ভিন্ন * __restrict__ dst,
        ধ্রুৱ int64_t ne00, ধ্রুৱ int64_t ne01, ধ্রুৱ int64_t ne02, ধ্রুৱ int64_t ne03,
        ধ্রুৱ int64_t s01, ধ্রুৱ int64_t s02, ধ্রুৱ int64_t s03,
        ধ্রুৱ int64_t  d1, ধ্রুৱ int64_t  d2, ধ্রুৱ int64_t  d3,
        ধ্রুৱ sycl::nd_item<3> & item, ভিন্ন * smem) {

    ধ্রুৱ পূর্ণাঙ্ক tid = item.get_local_id(2);
    ধ্রুৱ পূর্ণাঙ্ক block_size = item.get_local_range(2);
    ধ্রুৱ পূর্ণাঙ্ক lane = tid % WARP_SIZE;
    ধ্রুৱ পূর্ণাঙ্ক warp = tid / WARP_SIZE;
    ধ্রুৱ পূর্ণাঙ্ক warps_per_block = block_size / WARP_SIZE;

    ভিন্ন * s_vals      = smem;
    ভিন্ন * s_warp_sums = smem + block_size;
    ভিন্ন * s_carry     = smem + block_size + warps_per_block;

    যদি (tid == 0) {
        s_carry[0] = 0.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    ধ্রুৱ int64_t i3 = item.get_group(0);
    ধ্রুৱ int64_t i2 = item.get_group(1);
    ধ্রুৱ int64_t i1 = item.get_group(2);
    যদি (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        ঘূৰাই;
    }

    ধ্রুৱ ভিন্ন * src_row = src + i1 * s01 + i2 * s02 + i3 * s03;
    ভিন্ন       * dst_row = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr পূর্ণাঙ্ক অঙ্ক_unroll = 4;
    ভিন্ন temp[অঙ্ক_unroll];

    বাবে (int64_t i = 0; i < ne00; i += অঙ্ক_unroll * block_size) {
        int64_t idx = i + tid * অঙ্ক_unroll;

        temp[0] = (idx < ne00 ? src_row[idx] : 0.0f);
#pragma unroll
        বাবে (পূর্ণাঙ্ক j = 1; j < অঙ্ক_unroll; j++) {
            temp[j] = temp[j - 1];
            যদি (idx + j < ne00) {
                temp[j] += src_row[idx + j];
            }
        }

        ভিন্ন val = (idx < ne00) ? temp[অঙ্ক_unroll - 1] : 0.0f;

        val = warp_prefix_inclusive_sum_f32(val, item);
        s_vals[tid] = val;

        যদি (lane == WARP_SIZE - 1) {
            s_warp_sums[warp] = val;
        }
        item.barrier(sycl::access::fence_space::local_space);

        যদি (warp == 0) {
            ভিন্ন w = (tid < warps_per_block) ? s_warp_sums[tid] : 0.0f;
            ভিন্ন inc = warp_prefix_inclusive_sum_f32(w, item);
            যদি (tid < warps_per_block) {
                s_warp_sums[tid] = inc - w;
            }
            যদি (tid == warps_per_block - 1) {
                s_carry[1] = inc;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        ভিন্ন carry = s_carry[0];
        ভিন্ন final_offset = s_vals[tid] + s_warp_sums[warp] + carry - temp[অঙ্ক_unroll - 1];

#pragma unroll
        বাবে (পূর্ণাঙ্ক j = 0; j < অঙ্ক_unroll; j++) {
            যদি (idx + j < ne00) {
                dst_row[idx + j] = temp[j] + final_offset;
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        যদি (tid == 0) {
            s_carry[0] += s_carry[1];
        }
    }
}

অন্তঃস্থ শূন্য ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুৱ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ধ্রুৱ ভিন্ন * src_d = static_cast<ধ্রুৱ ভিন্ন *>(src0->data);
    ভিন্ন       * dst_d = static_cast<ভিন্ন *>(dst->data);

    ধ্রুৱ int64_t ne00 = src0->ne[0];
    ধ্রুৱ int64_t ne01 = src0->ne[1];
    ধ্রুৱ int64_t ne02 = src0->ne[2];
    ধ্রুৱ int64_t ne03 = src0->ne[3];

    ধ্রুৱ size_t ts = পরিমাণ(ভিন্ন);
    ধ্রুৱ int64_t s01 = src0->nb[1] / ts;
    ধ্রুৱ int64_t s02 = src0->nb[2] / ts;
    ধ্রুৱ int64_t s03 = src0->nb[3] / ts;
    ধ্রুৱ int64_t d1  = dst->nb[1] / ts;
    ধ্রুৱ int64_t d2  = dst->nb[2] / ts;
    ধ্রুৱ int64_t d3  = dst->nb[3] / ts;

    ধ্রুৱ পূর্ণাঙ্ক অঙ্ক_warps = (ne00 + WARP_SIZE - 1) / WARP_SIZE;
    পূর্ণাঙ্ক block_size = অঙ্ক_warps * WARP_SIZE;
    block_size = std::min(block_size, SYCL_CUMSUM_BLOCK_SIZE);
    ধ্রুৱ পূর্ণাঙ্ক warps_per_block = block_size / WARP_SIZE;
    ধ্রুৱ পূর্ণাঙ্ক smem_size = block_size + warps_per_block + 2;

    ধ্রুৱ sycl::range<3> grid(ne03, ne02, ne01);
    ধ্রুৱ sycl::range<3> block(1, 1, block_size);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<ভিন্ন, 1> smem_acc(sycl::range<1>(smem_size), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  item, get_pointer(smem_acc));
            });
    });
}

শূন্য ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cumsum(ctx, dst);
}
