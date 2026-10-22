#అన్తర్భావ "presets.hpp"
#అన్తర్భావ "common.hpp"
#అన్తర్భావ "ggml.h"
#అన్తర్భావ "set.hpp"
#అన్తర్భావ <cstdint>
#అన్తర్భావ <sycl/sycl.hpp>
ఉపయోగించి నామాకాశ sycl;

// Internal function: perform element-wise set operation for each thread
అంతఃవరుస శూన్య సమితి_f32(స్థిరరాశి దశాంశం* src, దశాంశం* dst,
                    స్థిరరాశి int64_t ne0, స్థిరరాశి int64_t ne1,
                    స్థిరరాశి int64_t ne2, స్థిరరాశి int64_t ne3,
                    స్థిరరాశి int64_t nb[3], స్థిరరాశి int64_t src_nb[3],
                    స్థిరరాశి int64_t స్థానభ్రంశం_elem,
                    స్థిరరాశి nd_అంశం<1>& అంశం)
{
    స్థిరరాశి size_t idx = అంశం.get_global_id(0);
    స్థిరరాశి size_t మొత్తం = ne0 * ne1 * ne2 * ne3;
    ఒకవేళ (idx >= మొత్తం) తిరిగి;

    // Convert linear index to 4D indices
    స్థిరరాశి size_t i3 = idx / (ne2 * ne1 * ne0);
    స్థిరరాశి size_t rem = idx % (ne2 * ne1 * ne0);
    స్థిరరాశి size_t i2 = rem / (ne1 * ne0);
    స్థిరరాశి size_t rem2 = rem % (ne1 * ne0);
    స్థిరరాశి size_t i1 = rem2 / ne0;
    స్థిరరాశి size_t i0 = rem2 % ne0;

    // Compute source and destination indices and copy
    dst[i0 + i1*nb[0] + i2*nb[1] + i3*nb[2] + స్థానభ్రంశం_elem] =
        src[i0 + i1*src_nb[0] + i2*src_nb[1] + i3*src_nb[2]];
}

// Main function: prepare GPU queue and launch parallel_for
శూన్య ggml_sycl_op_సమితి(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    స్థిరరాశి ggml_tensor* src0 = dst->src[0];
    స్థిరరాశి ggml_tensor* src1 = dst->src[1];

    // Ensure shapes and types are compatible
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(dst->type == src0->type && src0->type == src1->type && dst->type == GGML_TYPE_F32);

    స్థిరరాశి int32_t* opts = (స్థిరరాశి int32_t*) dst->op_params;
    స్థిరరాశి int64_t nb[3]     = {opts[0]/పరిమాణం(దశాంశం), opts[1]/పరిమాణం(దశాంశం), opts[2]/పరిమాణం(దశాంశం)};
    స్థిరరాశి int64_t స్థానభ్రంశం_elem = opts[3] / పరిమాణం(దశాంశం);
    స్థిరరాశి తార్కిక inplace = opts[4];

    దశాంశం* dst_ptr = (దశాంశం*) dst->దత్తాంశం;
    స్థిరరాశి దశాంశం* src0_ptr = (స్థిరరాశి దశాంశం*) src0->దత్తాంశం;
    స్థిరరాశి దశాంశం* src1_ptr = (స్థిరరాశి దశాంశం*) src1->దత్తాంశం;

    వరుస_ptr stream = ctx.stream();

    // Copy src0 to dst if not inplace
    ఒకవేళ (!inplace)
        stream->జ్ఞాపకం_నకలు(dst_ptr, src0_ptr, ggml_nbytes(dst));

    స్థిరరాశి int64_t ne[4] = {src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]};
    స్థిరరాశి int64_t src_nb[3] = {src1->nb[1]/పరిమాణం(దశాంశం), src1->nb[2]/పరిమాణం(దశాంశం), src1->nb[3]/పరిమాణం(దశాంశం)};

    స్థిరరాశి size_t మొత్తం_దారాలు = ne[0]*ne[1]*ne[2]*ne[3];
    స్థిరరాశి size_t grid_కొలత = ((మొత్తం_దారాలు + SYCL_సమితి_ఖండం_కొలత - 1) / SYCL_సమితి_ఖండం_కొలత) * SYCL_సమితి_ఖండం_కొలత;

    // Copy src0 to dst if not inplace
    stream->parallel_for(
        nd_range<1>(range<1>(grid_కొలత), range<1>(SYCL_సమితి_ఖండం_కొలత)),
        [=](nd_అంశం<1> అంశం) {
            సమితి_f32(src1_ptr, dst_ptr,
                ne[0], ne[1], ne[2], ne[3],
                nb, src_nb, స్థానభ్రంశం_elem, అంశం); }
    );
}
