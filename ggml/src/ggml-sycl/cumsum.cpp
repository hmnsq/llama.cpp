#ضمن "cumsum.hpp"
#ضمن "common.hpp"

#ضمن <algorithm>

#define SYCL_CUMSUM_BLOCK_SIZE 256

ساكن __dpct_inline__ عائم warp_prefix_inclusive_sum_f32(عائم x, ثابت sycl::nd_item<3> & item) {
    أرجع sycl::inclusive_scan_over_group(item.get_sub_group(), x, sycl::plus<عائم>());
}

ساكن فراغ cumsum_f32_kernel(
        ثابت عائم * __restrict__ src, عائم * __restrict__ dst,
        ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02, ثابت int64_t ne03,
        ثابت int64_t s01, ثابت int64_t s02, ثابت int64_t s03,
        ثابت int64_t  d1, ثابت int64_t  d2, ثابت int64_t  d3,
        ثابت sycl::nd_item<3> & item, عائم * smem) {

    ثابت صحيح tid = item.get_local_id(2);
    ثابت صحيح block_size = item.get_local_range(2);
    ثابت صحيح lane = tid % WARP_SIZE;
    ثابت صحيح warp = tid / WARP_SIZE;
    ثابت صحيح warps_per_block = block_size / WARP_SIZE;

    عائم * s_vals      = smem;
    عائم * s_warp_sums = smem + block_size;
    عائم * s_carry     = smem + block_size + warps_per_block;

    𐡄𐡍 (tid == 0) {
        s_carry[0] = 0.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    ثابت int64_t i3 = item.get_group(0);
    ثابت int64_t i2 = item.get_group(1);
    ثابت int64_t i1 = item.get_group(2);
    𐡄𐡍 (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        أرجع;
    }

    ثابت عائم * src_row = src + i1 * s01 + i2 * s02 + i3 * s03;
    عائم       * dst_row = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr صحيح num_unroll = 4;
    عائم temp[num_unroll];

    لكل (int64_t i = 0; i < ne00; i += num_unroll * block_size) {
        int64_t idx = i + tid * num_unroll;

        temp[0] = (idx < ne00 ? src_row[idx] : 0.0f);
#pragma unroll
        لكل (صحيح j = 1; j < num_unroll; j++) {
            temp[j] = temp[j - 1];
            𐡄𐡍 (idx + j < ne00) {
                temp[j] += src_row[idx + j];
            }
        }

        عائم val = (idx < ne00) ? temp[num_unroll - 1] : 0.0f;

        val = warp_prefix_inclusive_sum_f32(val, item);
        s_vals[tid] = val;

        𐡄𐡍 (lane == WARP_SIZE - 1) {
            s_warp_sums[warp] = val;
        }
        item.barrier(sycl::access::fence_space::local_space);

        𐡄𐡍 (warp == 0) {
            عائم w = (tid < warps_per_block) ? s_warp_sums[tid] : 0.0f;
            عائم inc = warp_prefix_inclusive_sum_f32(w, item);
            𐡄𐡍 (tid < warps_per_block) {
                s_warp_sums[tid] = inc - w;
            }
            𐡄𐡍 (tid == warps_per_block - 1) {
                s_carry[1] = inc;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        عائم carry = s_carry[0];
        عائم final_offset = s_vals[tid] + s_warp_sums[warp] + carry - temp[num_unroll - 1];

#pragma unroll
        لكل (صحيح j = 0; j < num_unroll; j++) {
            𐡄𐡍 (idx + j < ne00) {
                dst_row[idx + j] = temp[j] + final_offset;
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        𐡄𐡍 (tid == 0) {
            s_carry[0] += s_carry[1];
        }
    }
}

مضمن فراغ ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ثابت عائم * src_d = static_cast<ثابت عائم *>(src0->data);
    عائم       * dst_d = static_cast<عائم *>(dst->data);

    ثابت int64_t ne00 = src0->ne[0];
    ثابت int64_t ne01 = src0->ne[1];
    ثابت int64_t ne02 = src0->ne[2];
    ثابت int64_t ne03 = src0->ne[3];

    ثابت size_t ts = حجم(عائم);
    ثابت int64_t s01 = src0->nb[1] / ts;
    ثابت int64_t s02 = src0->nb[2] / ts;
    ثابت int64_t s03 = src0->nb[3] / ts;
    ثابت int64_t d1  = dst->nb[1] / ts;
    ثابت int64_t d2  = dst->nb[2] / ts;
    ثابت int64_t d3  = dst->nb[3] / ts;

    ثابت صحيح num_warps = (ne00 + WARP_SIZE - 1) / WARP_SIZE;
    صحيح block_size = num_warps * WARP_SIZE;
    block_size = std::min(block_size, SYCL_CUMSUM_BLOCK_SIZE);
    ثابت صحيح warps_per_block = block_size / WARP_SIZE;
    ثابت صحيح smem_size = block_size + warps_per_block + 2;

    ثابت sycl::range<3> grid(ne03, ne02, ne01);
    ثابت sycl::range<3> block(1, 1, block_size);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<عائم, 1> smem_acc(sycl::range<1>(smem_size), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  item, get_pointer(smem_acc));
            });
    });
}

فراغ ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cumsum(ctx, dst);
}
