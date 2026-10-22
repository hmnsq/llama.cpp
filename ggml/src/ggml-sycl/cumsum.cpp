#अन्तर्भाव "cumsum.hpp"
#अन्तर्भाव "common.hpp"

#अन्तर्भाव <algorithm>

#परिभाषय SYCL_CUMSUM_खण्डम्_आकार 256

स्थिर __dpct_inline__ भिन्न warp_उपसर्गः_inclusive_योग_f32(भिन्न x, ध्रुव sycl::nd_item<3> & item) {
    प्रत्यागम sycl::inclusive_अवलोकनम्_over_गणः(item.get_sub_गणः(), x, sycl::plus<भिन्न>());
}

स्थिर शून्य cumsum_f32_kernel(
        ध्रुव भिन्न * __restrict__ src, भिन्न * __restrict__ dst,
        ध्रुव int64_t ne00, ध्रुव int64_t ne01, ध्रुव int64_t ne02, ध्रुव int64_t ne03,
        ध्रुव int64_t s01, ध्रुव int64_t s02, ध्रुव int64_t s03,
        ध्रुव int64_t  d1, ध्रुव int64_t  d2, ध्रुव int64_t  d3,
        ध्रुव sycl::nd_item<3> & item, भिन्न * smem) {

    ध्रुव पूर्णाङ्क tid = item.get_local_id(2);
    ध्रुव पूर्णाङ्क खण्डम्_आकार_3 = item.get_local_range(2);
    ध्रुव पूर्णाङ्क lane = tid % WARP_आकार;
    ध्रुव पूर्णाङ्क warp = tid / WARP_आकार;
    ध्रुव पूर्णाङ्क warps_प्रत्येकम्_खण्डम् = खण्डम्_आकार_3 / WARP_आकार;

    भिन्न * s_vals      = smem;
    भिन्न * s_warp_sums = smem + खण्डम्_आकार_3;
    भिन्न * s_carry     = smem + खण्डम्_आकार_3 + warps_प्रत्येकम्_खण्डम्;

    यदि (tid == 0) {
        s_carry[0] = 0.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    ध्रुव int64_t i3 = item.get_गणः(0);
    ध्रुव int64_t i2 = item.get_गणः(1);
    ध्रुव int64_t i1 = item.get_गणः(2);
    यदि (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        प्रत्यागम;
    }

    ध्रुव भिन्न * src_row = src + i1 * s01 + i2 * s02 + i3 * s03;
    भिन्न       * dst_row = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr पूर्णाङ्क अङ्कः_unroll = 4;
    भिन्न अस्थायी[अङ्कः_unroll];

    प्रति (int64_t i = 0; i < ne00; i += अङ्कः_unroll * खण्डम्_आकार_3) {
        int64_t idx = i + tid * अङ्कः_unroll;

        अस्थायी[0] = (idx < ne00 ? src_row[idx] : 0.0f);
#निर्देशः unroll
        प्रति (पूर्णाङ्क j = 1; j < अङ्कः_unroll; j++) {
            अस्थायी[j] = अस्थायी[j - 1];
            यदि (idx + j < ne00) {
                अस्थायी[j] += src_row[idx + j];
            }
        }

        भिन्न val = (idx < ne00) ? अस्थायी[अङ्कः_unroll - 1] : 0.0f;

        val = warp_उपसर्गः_inclusive_योग_f32(val, item);
        s_vals[tid] = val;

        यदि (lane == WARP_आकार - 1) {
            s_warp_sums[warp] = val;
        }
        item.barrier(sycl::access::fence_space::local_space);

        यदि (warp == 0) {
            भिन्न w = (tid < warps_प्रत्येकम्_खण्डम्) ? s_warp_sums[tid] : 0.0f;
            भिन्न inc = warp_उपसर्गः_inclusive_योग_f32(w, item);
            यदि (tid < warps_प्रत्येकम्_खण्डम्) {
                s_warp_sums[tid] = inc - w;
            }
            यदि (tid == warps_प्रत्येकम्_खण्डम् - 1) {
                s_carry[1] = inc;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        भिन्न carry = s_carry[0];
        भिन्न final_विस्थापनम् = s_vals[tid] + s_warp_sums[warp] + carry - अस्थायी[अङ्कः_unroll - 1];

#निर्देशः unroll
        प्रति (पूर्णाङ्क j = 0; j < अङ्कः_unroll; j++) {
            यदि (idx + j < ne00) {
                dst_row[idx + j] = अस्थायी[j] + final_विस्थापनम्;
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
    SYCL_CHECK(ggml_sycl_समुच्चयः_device(ctx.device));

    ध्रुव भिन्न * src_d = static_cast<ध्रुव भिन्न *>(src0->दत्त);
    भिन्न       * dst_d = static_cast<भिन्न *>(dst->दत्त);

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

    ध्रुव पूर्णाङ्क अङ्कः_warps = (ne00 + WARP_आकार - 1) / WARP_आकार;
    पूर्णाङ्क खण्डम्_आकार_3 = अङ्कः_warps * WARP_आकार;
    खण्डम्_आकार_3 = std::min(खण्डम्_आकार_3, SYCL_CUMSUM_खण्डम्_आकार);
    ध्रुव पूर्णाङ्क warps_प्रत्येकम्_खण्डम् = खण्डम्_आकार_3 / WARP_आकार;
    ध्रुव पूर्णाङ्क smem_आकार = खण्डम्_आकार_3 + warps_प्रत्येकम्_खण्डम् + 2;

    ध्रुव sycl::range<3> grid(ne03, ne02, ne01);
    ध्रुव sycl::range<3> खण्डम्(1, 1, खण्डम्_आकार_3);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<भिन्न, 1> smem_acc(sycl::range<1>(smem_आकार), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * खण्डम्, खण्डम्),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
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
