#ਅਨ੍ਤਰ੍ਭਾਵ "presets.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "set.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/sycl.hpp>
ਵਰਤੋਂ ਨਾਮਾਕਾਸ਼ sycl;

// Internal function: perform element-wise set operation for each thread
ਸਤਰਅੰਦਰ ਖਾਲੀ ਸੈੱਟ_f32(ਅਚਲ ਦਸ਼ਮਲਵ* src, ਦਸ਼ਮਲਵ* dst,
                    ਅਚਲ int64_t ne0, ਅਚਲ int64_t ne1,
                    ਅਚਲ int64_t ne2, ਅਚਲ int64_t ne3,
                    ਅਚਲ int64_t nb[3], ਅਚਲ int64_t src_nb[3],
                    ਅਚਲ int64_t ਵਿੱਥ_elem,
                    ਅਚਲ nd_ਵਸਤੂ<1>& ਵਸਤੂ)
{
    ਅਚਲ size_t idx = ਵਸਤੂ.get_global_id(0);
    ਅਚਲ size_t ਕੁੱਲ = ne0 * ne1 * ne2 * ne3;
    ਜੇ (idx >= ਕੁੱਲ) ਵਾਪਸ;

    // Convert linear index to 4D indices
    ਅਚਲ size_t i3 = idx / (ne2 * ne1 * ne0);
    ਅਚਲ size_t rem = idx % (ne2 * ne1 * ne0);
    ਅਚਲ size_t i2 = rem / (ne1 * ne0);
    ਅਚਲ size_t rem2 = rem % (ne1 * ne0);
    ਅਚਲ size_t i1 = rem2 / ne0;
    ਅਚਲ size_t i0 = rem2 % ne0;

    // Compute source and destination indices and copy
    dst[i0 + i1*nb[0] + i2*nb[1] + i3*nb[2] + ਵਿੱਥ_elem] =
        src[i0 + i1*src_nb[0] + i2*src_nb[1] + i3*src_nb[2]];
}

// Main function: prepare GPU queue and launch parallel_for
ਖਾਲੀ ggml_sycl_op_ਸੈੱਟ(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ਅਚਲ ggml_tensor* src0 = dst->src[0];
    ਅਚਲ ggml_tensor* src1 = dst->src[1];

    // Ensure shapes and types are compatible
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(dst->type == src0->type && src0->type == src1->type && dst->type == GGML_TYPE_F32);

    ਅਚਲ int32_t* opts = (ਅਚਲ int32_t*) dst->op_params;
    ਅਚਲ int64_t nb[3]     = {opts[0]/ਆਕਾਰ(ਦਸ਼ਮਲਵ), opts[1]/ਆਕਾਰ(ਦਸ਼ਮਲਵ), opts[2]/ਆਕਾਰ(ਦਸ਼ਮਲਵ)};
    ਅਚਲ int64_t ਵਿੱਥ_elem = opts[3] / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ ਤਾਰਕਿਕ inplace = opts[4];

    ਦਸ਼ਮਲਵ* dst_ptr = (ਦਸ਼ਮਲਵ*) dst->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ* src0_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ*) src0->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ* src1_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ*) src1->ਅੰਕੜੇ;

    ਕਤਾਰ_ptr stream = ctx.stream();

    // Copy src0 to dst if not inplace
    ਜੇ (!inplace)
        stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_ptr, src0_ptr, ggml_nbytes(dst));

    ਅਚਲ int64_t ne[4] = {src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]};
    ਅਚਲ int64_t src_nb[3] = {src1->nb[1]/ਆਕਾਰ(ਦਸ਼ਮਲਵ), src1->nb[2]/ਆਕਾਰ(ਦਸ਼ਮਲਵ), src1->nb[3]/ਆਕਾਰ(ਦਸ਼ਮਲਵ)};

    ਅਚਲ size_t ਕੁੱਲ_ਧਾਗੇ = ne[0]*ne[1]*ne[2]*ne[3];
    ਅਚਲ size_t grid_ਮਾਪ = ((ਕੁੱਲ_ਧਾਗੇ + SYCL_ਸੈੱਟ_ਖੰਡ_ਮਾਪ - 1) / SYCL_ਸੈੱਟ_ਖੰਡ_ਮਾਪ) * SYCL_ਸੈੱਟ_ਖੰਡ_ਮਾਪ;

    // Copy src0 to dst if not inplace
    stream->parallel_for(
        nd_range<1>(range<1>(grid_ਮਾਪ), range<1>(SYCL_ਸੈੱਟ_ਖੰਡ_ਮਾਪ)),
        [=](nd_ਵਸਤੂ<1> ਵਸਤੂ) {
            ਸੈੱਟ_f32(src1_ptr, dst_ptr,
                ne[0], ne[1], ne[2], ne[3],
                nb, src_nb, ਵਿੱਥ_elem, ਵਸਤੂ); }
    );
}
