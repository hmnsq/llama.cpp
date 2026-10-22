#அந்தர்பாவ "cumsum.hpp"
#அந்தர்பாவ "common.hpp"

#அந்தர்பாவ <algorithm>

#define SYCL_CUMSUM_BLOCK_SIZE 256

நிலையான __dpct_inline__ தசம warp_prefix_inclusive_sum_f32(தசம x, மாறிலி sycl::nd_item<3> & item) {
    திரும்பு sycl::inclusive_scan_over_group(item.get_sub_group(), x, sycl::plus<தசம>());
}

நிலையான வெறுமை cumsum_f32_kernel(
        மாறிலி தசம * __restrict__ src, தசம * __restrict__ dst,
        மாறிலி int64_t ne00, மாறிலி int64_t ne01, மாறிலி int64_t ne02, மாறிலி int64_t ne03,
        மாறிலி int64_t s01, மாறிலி int64_t s02, மாறிலி int64_t s03,
        மாறிலி int64_t  d1, மாறிலி int64_t  d2, மாறிலி int64_t  d3,
        மாறிலி sycl::nd_item<3> & item, தசம * smem) {

    மாறிலி முழு tid = item.get_local_id(2);
    மாறிலி முழு block_size = item.get_local_range(2);
    மாறிலி முழு lane = tid % WARP_SIZE;
    மாறிலி முழு warp = tid / WARP_SIZE;
    மாறிலி முழு warps_per_block = block_size / WARP_SIZE;

    தசம * s_vals      = smem;
    தசம * s_warp_sums = smem + block_size;
    தசம * s_carry     = smem + block_size + warps_per_block;

    எனில் (tid == 0) {
        s_carry[0] = 0.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    மாறிலி int64_t i3 = item.get_group(0);
    மாறிலி int64_t i2 = item.get_group(1);
    மாறிலி int64_t i1 = item.get_group(2);
    எனில் (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        திரும்பு;
    }

    மாறிலி தசம * src_row = src + i1 * s01 + i2 * s02 + i3 * s03;
    தசம       * dst_row = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr முழு இலக்கம்_unroll = 4;
    தசம temp[இலக்கம்_unroll];

    க்கு (int64_t i = 0; i < ne00; i += இலக்கம்_unroll * block_size) {
        int64_t idx = i + tid * இலக்கம்_unroll;

        temp[0] = (idx < ne00 ? src_row[idx] : 0.0f);
#pragma unroll
        க்கு (முழு j = 1; j < இலக்கம்_unroll; j++) {
            temp[j] = temp[j - 1];
            எனில் (idx + j < ne00) {
                temp[j] += src_row[idx + j];
            }
        }

        தசம val = (idx < ne00) ? temp[இலக்கம்_unroll - 1] : 0.0f;

        val = warp_prefix_inclusive_sum_f32(val, item);
        s_vals[tid] = val;

        எனில் (lane == WARP_SIZE - 1) {
            s_warp_sums[warp] = val;
        }
        item.barrier(sycl::access::fence_space::local_space);

        எனில் (warp == 0) {
            தசம w = (tid < warps_per_block) ? s_warp_sums[tid] : 0.0f;
            தசம inc = warp_prefix_inclusive_sum_f32(w, item);
            எனில் (tid < warps_per_block) {
                s_warp_sums[tid] = inc - w;
            }
            எனில் (tid == warps_per_block - 1) {
                s_carry[1] = inc;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        தசம carry = s_carry[0];
        தசம final_offset = s_vals[tid] + s_warp_sums[warp] + carry - temp[இலக்கம்_unroll - 1];

#pragma unroll
        க்கு (முழு j = 0; j < இலக்கம்_unroll; j++) {
            எனில் (idx + j < ne00) {
                dst_row[idx + j] = temp[j] + final_offset;
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        எனில் (tid == 0) {
            s_carry[0] += s_carry[1];
        }
    }
}

உள்வரி வெறுமை ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    மாறிலி தசம * src_d = static_cast<மாறிலி தசம *>(src0->data);
    தசம       * dst_d = static_cast<தசம *>(dst->data);

    மாறிலி int64_t ne00 = src0->ne[0];
    மாறிலி int64_t ne01 = src0->ne[1];
    மாறிலி int64_t ne02 = src0->ne[2];
    மாறிலி int64_t ne03 = src0->ne[3];

    மாறிலி size_t ts = அளவு(தசம);
    மாறிலி int64_t s01 = src0->nb[1] / ts;
    மாறிலி int64_t s02 = src0->nb[2] / ts;
    மாறிலி int64_t s03 = src0->nb[3] / ts;
    மாறிலி int64_t d1  = dst->nb[1] / ts;
    மாறிலி int64_t d2  = dst->nb[2] / ts;
    மாறிலி int64_t d3  = dst->nb[3] / ts;

    மாறிலி முழு இலக்கம்_warps = (ne00 + WARP_SIZE - 1) / WARP_SIZE;
    முழு block_size = இலக்கம்_warps * WARP_SIZE;
    block_size = std::min(block_size, SYCL_CUMSUM_BLOCK_SIZE);
    மாறிலி முழு warps_per_block = block_size / WARP_SIZE;
    மாறிலி முழு smem_size = block_size + warps_per_block + 2;

    மாறிலி sycl::range<3> grid(ne03, ne02, ne01);
    மாறிலி sycl::range<3> block(1, 1, block_size);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<தசம, 1> smem_acc(sycl::range<1>(smem_size), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  item, get_pointer(smem_acc));
            });
    });
}

வெறுமை ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cumsum(ctx, dst);
}
