#شامل "presets.hpp"
#شامل "common.hpp"
#شامل "ggml.h"
#شامل "set.hpp"
#شامل <cstdint>
#شامل <sycl/sycl.hpp>
استعمال نامخلا sycl;

// Internal function: perform element-wise set operation for each thread
اندرونی خالی سیٹ_f32(ثابت اعشاری* src, اعشاری* dst,
                    ثابت int64_t ne0, ثابت int64_t ne1,
                    ثابت int64_t ne2, ثابت int64_t ne3,
                    ثابت int64_t nb[3], ثابت int64_t src_nb[3],
                    ثابت int64_t انحراف_elem,
                    ثابت nd_شے<1>& شے)
{
    ثابت size_t idx = شے.get_global_id(0);
    ثابت size_t کل = ne0 * ne1 * ne2 * ne3;
    اگر (idx >= کل) واپس;

    // Convert linear index to 4D indices
    ثابت size_t i3 = idx / (ne2 * ne1 * ne0);
    ثابت size_t rem = idx % (ne2 * ne1 * ne0);
    ثابت size_t i2 = rem / (ne1 * ne0);
    ثابت size_t rem2 = rem % (ne1 * ne0);
    ثابت size_t i1 = rem2 / ne0;
    ثابت size_t i0 = rem2 % ne0;

    // Compute source and destination indices and copy
    dst[i0 + i1*nb[0] + i2*nb[1] + i3*nb[2] + انحراف_elem] =
        src[i0 + i1*src_nb[0] + i2*src_nb[1] + i3*src_nb[2]];
}

// Main function: prepare GPU queue and launch parallel_for
خالی ggml_sycl_op_سیٹ(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ثابت ggml_tensor* src0 = dst->src[0];
    ثابت ggml_tensor* src1 = dst->src[1];

    // Ensure shapes and types are compatible
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(dst->type == src0->type && src0->type == src1->type && dst->type == GGML_TYPE_F32);

    ثابت int32_t* opts = (ثابت int32_t*) dst->op_params;
    ثابت int64_t nb[3]     = {opts[0]/حجم(اعشاری), opts[1]/حجم(اعشاری), opts[2]/حجم(اعشاری)};
    ثابت int64_t انحراف_elem = opts[3] / حجم(اعشاری);
    ثابت منطقی inplace = opts[4];

    اعشاری* dst_ptr = (اعشاری*) dst->ڈیٹا;
    ثابت اعشاری* src0_ptr = (ثابت اعشاری*) src0->ڈیٹا;
    ثابت اعشاری* src1_ptr = (ثابت اعشاری*) src1->ڈیٹا;

    قطار_ptr stream = ctx.stream();

    // Copy src0 to dst if not inplace
    اگر (!inplace)
        stream->حافظہ_نقل(dst_ptr, src0_ptr, ggml_nbytes(dst));

    ثابت int64_t ne[4] = {src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]};
    ثابت int64_t src_nb[3] = {src1->nb[1]/حجم(اعشاری), src1->nb[2]/حجم(اعشاری), src1->nb[3]/حجم(اعشاری)};

    ثابت size_t کل_دھاگے = ne[0]*ne[1]*ne[2]*ne[3];
    ثابت size_t grid_سائز = ((کل_دھاگے + SYCL_سیٹ_بلاک_سائز - 1) / SYCL_سیٹ_بلاک_سائز) * SYCL_سیٹ_بلاک_سائز;

    // Copy src0 to dst if not inplace
    stream->parallel_for(
        nd_range<1>(range<1>(grid_سائز), range<1>(SYCL_سیٹ_بلاک_سائز)),
        [=](nd_شے<1> شے) {
            سیٹ_f32(src1_ptr, dst_ptr,
                ne[0], ne[1], ne[2], ne[3],
                nb, src_nb, انحراف_elem, شے); }
    );
}
