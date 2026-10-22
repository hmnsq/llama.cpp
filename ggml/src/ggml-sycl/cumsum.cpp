#شامل "cumsum.hpp"
#شامل "common.hpp"

#شامل <algorithm>

#define SYCL_CUMSUM_BLOCK_SIZE 256

ایستا __dpct_inline__ اعشاری warp_prefix_inclusive_sum_f32(اعشاری x, ثابت sycl::nd_item<3> & item) {
    بازگشت sycl::inclusive_scan_over_group(item.get_sub_group(), x, sycl::plus<اعشاری>());
}

ایستا تهی cumsum_f32_kernel(
        ثابت اعشاری * __restrict__ src, اعشاری * __restrict__ dst,
        ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02, ثابت int64_t ne03,
        ثابت int64_t s01, ثابت int64_t s02, ثابت int64_t s03,
        ثابت int64_t  d1, ثابت int64_t  d2, ثابت int64_t  d3,
        ثابت sycl::nd_item<3> & item, اعشاری * smem) {

    ثابت صحیح tid = item.get_local_id(2);
    ثابت صحیح block_size = item.get_local_range(2);
    ثابت صحیح lane = tid % WARP_SIZE;
    ثابت صحیح warp = tid / WARP_SIZE;
    ثابت صحیح warps_per_block = block_size / WARP_SIZE;

    اعشاری * s_vals      = smem;
    اعشاری * s_warp_sums = smem + block_size;
    اعشاری * s_carry     = smem + block_size + warps_per_block;

    اگر (tid == 0) {
        s_carry[0] = 0.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    ثابت int64_t i3 = item.get_group(0);
    ثابت int64_t i2 = item.get_group(1);
    ثابت int64_t i1 = item.get_group(2);
    اگر (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        بازگشت;
    }

    ثابت اعشاری * src_row = src + i1 * s01 + i2 * s02 + i3 * s03;
    اعشاری       * dst_row = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr صحیح رقم_unroll = 4;
    اعشاری temp[رقم_unroll];

    برای (int64_t i = 0; i < ne00; i += رقم_unroll * block_size) {
        int64_t idx = i + tid * رقم_unroll;

        temp[0] = (idx < ne00 ? src_row[idx] : 0.0f);
#pragma unroll
        برای (صحیح j = 1; j < رقم_unroll; j++) {
            temp[j] = temp[j - 1];
            اگر (idx + j < ne00) {
                temp[j] += src_row[idx + j];
            }
        }

        اعشاری val = (idx < ne00) ? temp[رقم_unroll - 1] : 0.0f;

        val = warp_prefix_inclusive_sum_f32(val, item);
        s_vals[tid] = val;

        اگر (lane == WARP_SIZE - 1) {
            s_warp_sums[warp] = val;
        }
        item.barrier(sycl::access::fence_space::local_space);

        اگر (warp == 0) {
            اعشاری w = (tid < warps_per_block) ? s_warp_sums[tid] : 0.0f;
            اعشاری inc = warp_prefix_inclusive_sum_f32(w, item);
            اگر (tid < warps_per_block) {
                s_warp_sums[tid] = inc - w;
            }
            اگر (tid == warps_per_block - 1) {
                s_carry[1] = inc;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        اعشاری carry = s_carry[0];
        اعشاری final_offset = s_vals[tid] + s_warp_sums[warp] + carry - temp[رقم_unroll - 1];

#pragma unroll
        برای (صحیح j = 0; j < رقم_unroll; j++) {
            اگر (idx + j < ne00) {
                dst_row[idx + j] = temp[j] + final_offset;
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        اگر (tid == 0) {
            s_carry[0] += s_carry[1];
        }
    }
}

درونخطی تهی ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ثابت اعشاری * src_d = static_cast<ثابت اعشاری *>(src0->data);
    اعشاری       * dst_d = static_cast<اعشاری *>(dst->data);

    ثابت int64_t ne00 = src0->ne[0];
    ثابت int64_t ne01 = src0->ne[1];
    ثابت int64_t ne02 = src0->ne[2];
    ثابت int64_t ne03 = src0->ne[3];

    ثابت size_t ts = اندازه(اعشاری);
    ثابت int64_t s01 = src0->nb[1] / ts;
    ثابت int64_t s02 = src0->nb[2] / ts;
    ثابت int64_t s03 = src0->nb[3] / ts;
    ثابت int64_t d1  = dst->nb[1] / ts;
    ثابت int64_t d2  = dst->nb[2] / ts;
    ثابت int64_t d3  = dst->nb[3] / ts;

    ثابت صحیح رقم_warps = (ne00 + WARP_SIZE - 1) / WARP_SIZE;
    صحیح block_size = رقم_warps * WARP_SIZE;
    block_size = std::min(block_size, SYCL_CUMSUM_BLOCK_SIZE);
    ثابت صحیح warps_per_block = block_size / WARP_SIZE;
    ثابت صحیح smem_size = block_size + warps_per_block + 2;

    ثابت sycl::range<3> grid(ne03, ne02, ne01);
    ثابت sycl::range<3> block(1, 1, block_size);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<اعشاری, 1> smem_acc(sycl::range<1>(smem_size), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  item, get_pointer(smem_acc));
            });
    });
}

تهی ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cumsum(ctx, dst);
}
