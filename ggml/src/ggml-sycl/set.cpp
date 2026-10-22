#ಅನ್ತರ್ಭಾವ "presets.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "set.hpp"
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>
ಬಳಸಿ ನಾಮಾಕಾಶ sycl;

// Internal function: perform element-wise set operation for each thread
ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಗಣ_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* src, ದಶಮಾಂಶ* dst,
                    ಸ್ಥಿರಾಂಕ int64_t ne0, ಸ್ಥಿರಾಂಕ int64_t ne1,
                    ಸ್ಥಿರಾಂಕ int64_t ne2, ಸ್ಥಿರಾಂಕ int64_t ne3,
                    ಸ್ಥಿರಾಂಕ int64_t nb[3], ಸ್ಥಿರಾಂಕ int64_t src_nb[3],
                    ಸ್ಥಿರಾಂಕ int64_t ಸ್ಥಾನಾಂತರ_elem,
                    ಸ್ಥಿರಾಂಕ nd_ಅಂಶ<1>& ಅಂಶ)
{
    ಸ್ಥಿರಾಂಕ size_t idx = ಅಂಶ.get_global_id(0);
    ಸ್ಥಿರಾಂಕ size_t ಒಟ್ಟು = ne0 * ne1 * ne2 * ne3;
    ಆದರೆ (idx >= ಒಟ್ಟು) ಹಿಂತಿರುಗಿ;

    // Convert linear index to 4D indices
    ಸ್ಥಿರಾಂಕ size_t i3 = idx / (ne2 * ne1 * ne0);
    ಸ್ಥಿರಾಂಕ size_t rem = idx % (ne2 * ne1 * ne0);
    ಸ್ಥಿರಾಂಕ size_t i2 = rem / (ne1 * ne0);
    ಸ್ಥಿರಾಂಕ size_t rem2 = rem % (ne1 * ne0);
    ಸ್ಥಿರಾಂಕ size_t i1 = rem2 / ne0;
    ಸ್ಥಿರಾಂಕ size_t i0 = rem2 % ne0;

    // Compute source and destination indices and copy
    dst[i0 + i1*nb[0] + i2*nb[1] + i3*nb[2] + ಸ್ಥಾನಾಂತರ_elem] =
        src[i0 + i1*src_nb[0] + i2*src_nb[1] + i3*src_nb[2]];
}

// Main function: prepare GPU queue and launch parallel_for
ಶೂನ್ಯ ggml_sycl_op_ಗಣ(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor* src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor* src1 = dst->src[1];

    // Ensure shapes and types are compatible
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(dst->type == src0->type && src0->type == src1->type && dst->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ int32_t* opts = (ಸ್ಥಿರಾಂಕ int32_t*) dst->op_params;
    ಸ್ಥಿರಾಂಕ int64_t nb[3]     = {opts[0]/ಗಾತ್ರ(ದಶಮಾಂಶ), opts[1]/ಗಾತ್ರ(ದಶಮಾಂಶ), opts[2]/ಗಾತ್ರ(ದಶಮಾಂಶ)};
    ಸ್ಥಿರಾಂಕ int64_t ಸ್ಥಾನಾಂತರ_elem = opts[3] / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ inplace = opts[4];

    ದಶಮಾಂಶ* dst_ptr = (ದಶಮಾಂಶ*) dst->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* src0_ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*) src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* src1_ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*) src1->ದತ್ತಾಂಶ;

    ಸರತಿ_ptr stream = ctx.stream();

    // Copy src0 to dst if not inplace
    ಆದರೆ (!inplace)
        stream->ಸ್ಮರಣೆ_ನಕಲು(dst_ptr, src0_ptr, ggml_nbytes(dst));

    ಸ್ಥಿರಾಂಕ int64_t ne[4] = {src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]};
    ಸ್ಥಿರಾಂಕ int64_t src_nb[3] = {src1->nb[1]/ಗಾತ್ರ(ದಶಮಾಂಶ), src1->nb[2]/ಗಾತ್ರ(ದಶಮಾಂಶ), src1->nb[3]/ಗಾತ್ರ(ದಶಮಾಂಶ)};

    ಸ್ಥಿರಾಂಕ size_t ಒಟ್ಟು_ಎಳೆಗಳು = ne[0]*ne[1]*ne[2]*ne[3];
    ಸ್ಥಿರಾಂಕ size_t grid_ಗಾತ್ರ = ((ಒಟ್ಟು_ಎಳೆಗಳು + SYCL_ಗಣ_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_ಗಣ_ಖಂಡ_ಗಾತ್ರ) * SYCL_ಗಣ_ಖಂಡ_ಗಾತ್ರ;

    // Copy src0 to dst if not inplace
    stream->parallel_for(
        nd_range<1>(range<1>(grid_ಗಾತ್ರ), range<1>(SYCL_ಗಣ_ಖಂಡ_ಗಾತ್ರ)),
        [=](nd_ಅಂಶ<1> ಅಂಶ) {
            ಗಣ_f32(src1_ptr, dst_ptr,
                ne[0], ne[1], ne[2], ne[3],
                nb, src_nb, ಸ್ಥಾನಾಂತರ_elem, ಅಂಶ); }
    );
}
