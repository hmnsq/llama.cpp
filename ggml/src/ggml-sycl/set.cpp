#അന്തര്ഭാവ "presets.hpp"
#അന്തര്ഭാവ "common.hpp"
#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "set.hpp"
#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <sycl/sycl.hpp>
using നാമാകാശ sycl;

// Internal function: perform element-wise set operation for each thread
വരിയിൽ ശൂന്യം set_f32(സ്ഥിരം ദശാംശം* src, ദശാംശം* dst,
                    സ്ഥിരം int64_t ne0, സ്ഥിരം int64_t ne1,
                    സ്ഥിരം int64_t ne2, സ്ഥിരം int64_t ne3,
                    സ്ഥിരം int64_t nb[3], സ്ഥിരം int64_t src_nb[3],
                    സ്ഥിരം int64_t offset_elem,
                    സ്ഥിരം nd_item<1>& item)
{
    സ്ഥിരം size_t idx = item.get_global_id(0);
    സ്ഥിരം size_t total = ne0 * ne1 * ne2 * ne3;
    എങ്കിൽ (idx >= total) തിരികെ;

    // Convert linear index to 4D indices
    സ്ഥിരം size_t i3 = idx / (ne2 * ne1 * ne0);
    സ്ഥിരം size_t rem = idx % (ne2 * ne1 * ne0);
    സ്ഥിരം size_t i2 = rem / (ne1 * ne0);
    സ്ഥിരം size_t rem2 = rem % (ne1 * ne0);
    സ്ഥിരം size_t i1 = rem2 / ne0;
    സ്ഥിരം size_t i0 = rem2 % ne0;

    // Compute source and destination indices and copy
    dst[i0 + i1*nb[0] + i2*nb[1] + i3*nb[2] + offset_elem] =
        src[i0 + i1*src_nb[0] + i2*src_nb[1] + i3*src_nb[2]];
}

// Main function: prepare GPU queue and launch parallel_for
ശൂന്യം ggml_sycl_op_set(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    സ്ഥിരം ggml_tensor* src0 = dst->src[0];
    സ്ഥിരം ggml_tensor* src1 = dst->src[1];

    // Ensure shapes and types are compatible
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(dst->type == src0->type && src0->type == src1->type && dst->type == GGML_TYPE_F32);

    സ്ഥിരം int32_t* opts = (സ്ഥിരം int32_t*) dst->op_params;
    സ്ഥിരം int64_t nb[3]     = {opts[0]/വലുപ്പം(ദശാംശം), opts[1]/വലുപ്പം(ദശാംശം), opts[2]/വലുപ്പം(ദശാംശം)};
    സ്ഥിരം int64_t offset_elem = opts[3] / വലുപ്പം(ദശാംശം);
    സ്ഥിരം തർക്കം inplace = opts[4];

    ദശാംശം* dst_ptr = (ദശാംശം*) dst->data;
    സ്ഥിരം ദശാംശം* src0_ptr = (സ്ഥിരം ദശാംശം*) src0->data;
    സ്ഥിരം ദശാംശം* src1_ptr = (സ്ഥിരം ദശാംശം*) src1->data;

    queue_ptr stream = ctx.stream();

    // Copy src0 to dst if not inplace
    എങ്കിൽ (!inplace)
        stream->memcpy(dst_ptr, src0_ptr, ggml_nbytes(dst));

    സ്ഥിരം int64_t ne[4] = {src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]};
    സ്ഥിരം int64_t src_nb[3] = {src1->nb[1]/വലുപ്പം(ദശാംശം), src1->nb[2]/വലുപ്പം(ദശാംശം), src1->nb[3]/വലുപ്പം(ദശാംശം)};

    സ്ഥിരം size_t total_threads = ne[0]*ne[1]*ne[2]*ne[3];
    സ്ഥിരം size_t grid_size = ((total_threads + SYCL_SET_BLOCK_SIZE - 1) / SYCL_SET_BLOCK_SIZE) * SYCL_SET_BLOCK_SIZE;

    // Copy src0 to dst if not inplace
    stream->parallel_for(
        nd_range<1>(range<1>(grid_size), range<1>(SYCL_SET_BLOCK_SIZE)),
        [=](nd_item<1> item) {
            set_f32(src1_ptr, dst_ptr,
                ne[0], ne[1], ne[2], ne[3],
                nb, src_nb, offset_elem, item); }
    );
}
