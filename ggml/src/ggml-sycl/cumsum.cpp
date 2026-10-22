#समावेश "cumsum.hpp"
#समावेश "common.hpp"

#समावेश <algorithm>

#घोषणा SYCL_CUMSUM_BLOCK_आकार 256

अटल __dpct_inline__ भग्न warp_prefix_inclusive_योग_f32(भग्न x, स्थिर sycl::nd_item<3> & item) {
    वापस sycl::inclusive_scan_over_group(item.get_sub_group(), x, sycl::plus<भग्न>());
}

अटल व्योम cumsum_f32_kernel(
        स्थिर भग्न * __restrict__ src, भग्न * __restrict__ dst,
        स्थिर int64_t ne00, स्थिर int64_t ne01, स्थिर int64_t ne02, स्थिर int64_t ne03,
        स्थिर int64_t s01, स्थिर int64_t s02, स्थिर int64_t s03,
        स्थिर int64_t  d1, स्थिर int64_t  d2, स्थिर int64_t  d3,
        स्थिर sycl::nd_item<3> & item, भग्न * smem) {

    स्थिर पूर्णांक tid = item.get_local_id(2);
    स्थिर पूर्णांक block_आकार_2 = item.get_local_range(2);
    स्थिर पूर्णांक lane = tid % WARP_आकार;
    स्थिर पूर्णांक warp = tid / WARP_आकार;
    स्थिर पूर्णांक warps_per_block = block_आकार_2 / WARP_आकार;

    भग्न * s_vals      = smem;
    भग्न * s_warp_sums = smem + block_आकार_2;
    भग्न * s_carry     = smem + block_आकार_2 + warps_per_block;

    अगर (tid == 0) {
        s_carry[0] = 0.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    स्थिर int64_t i3 = item.get_group(0);
    स्थिर int64_t i2 = item.get_group(1);
    स्थिर int64_t i1 = item.get_group(2);
    अगर (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        वापस;
    }

    स्थिर भग्न * src_row = src + i1 * s01 + i2 * s02 + i3 * s03;
    भग्न       * dst_row = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr पूर्णांक अंक_unroll = 4;
    भग्न अस्थायी[अंक_unroll];

    क्रम (int64_t i = 0; i < ne00; i += अंक_unroll * block_आकार_2) {
        int64_t idx = i + tid * अंक_unroll;

        अस्थायी[0] = (idx < ne00 ? src_row[idx] : 0.0f);
#आशय unroll
        क्रम (पूर्णांक j = 1; j < अंक_unroll; j++) {
            अस्थायी[j] = अस्थायी[j - 1];
            अगर (idx + j < ne00) {
                अस्थायी[j] += src_row[idx + j];
            }
        }

        भग्न val = (idx < ne00) ? अस्थायी[अंक_unroll - 1] : 0.0f;

        val = warp_prefix_inclusive_योग_f32(val, item);
        s_vals[tid] = val;

        अगर (lane == WARP_आकार - 1) {
            s_warp_sums[warp] = val;
        }
        item.barrier(sycl::access::fence_space::local_space);

        अगर (warp == 0) {
            भग्न w = (tid < warps_per_block) ? s_warp_sums[tid] : 0.0f;
            भग्न inc = warp_prefix_inclusive_योग_f32(w, item);
            अगर (tid < warps_per_block) {
                s_warp_sums[tid] = inc - w;
            }
            अगर (tid == warps_per_block - 1) {
                s_carry[1] = inc;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        भग्न carry = s_carry[0];
        भग्न final_offset = s_vals[tid] + s_warp_sums[warp] + carry - अस्थायी[अंक_unroll - 1];

#आशय unroll
        क्रम (पूर्णांक j = 0; j < अंक_unroll; j++) {
            अगर (idx + j < ne00) {
                dst_row[idx + j] = अस्थायी[j] + final_offset;
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        अगर (tid == 0) {
            s_carry[0] += s_carry[1];
        }
    }
}

अंतरभूत व्योम ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    स्थिर भग्न * src_d = static_cast<स्थिर भग्न *>(src0->तथ्य);
    भग्न       * dst_d = static_cast<भग्न *>(dst->तथ्य);

    स्थिर int64_t ne00 = src0->ne[0];
    स्थिर int64_t ne01 = src0->ne[1];
    स्थिर int64_t ne02 = src0->ne[2];
    स्थिर int64_t ne03 = src0->ne[3];

    स्थिर माप_प्रकार ts = माप(भग्न);
    स्थिर int64_t s01 = src0->nb[1] / ts;
    स्थिर int64_t s02 = src0->nb[2] / ts;
    स्थिर int64_t s03 = src0->nb[3] / ts;
    स्थिर int64_t d1  = dst->nb[1] / ts;
    स्थिर int64_t d2  = dst->nb[2] / ts;
    स्थिर int64_t d3  = dst->nb[3] / ts;

    स्थिर पूर्णांक अंक_warps = (ne00 + WARP_आकार - 1) / WARP_आकार;
    पूर्णांक block_आकार_2 = अंक_warps * WARP_आकार;
    block_आकार_2 = std::min(block_आकार_2, SYCL_CUMSUM_BLOCK_आकार);
    स्थिर पूर्णांक warps_per_block = block_आकार_2 / WARP_आकार;
    स्थिर पूर्णांक smem_आकार = block_आकार_2 + warps_per_block + 2;

    स्थिर sycl::range<3> grid(ne03, ne02, ne01);
    स्थिर sycl::range<3> block(1, 1, block_आकार_2);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<भग्न, 1> smem_acc(sycl::range<1>(smem_आकार), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  item, get_pointer(smem_acc));
            });
    });
}

व्योम ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cumsum(ctx, dst);
}
