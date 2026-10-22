#अन्तर्भाव "presets.hpp"
#अन्तर्भाव "common.hpp"
#अन्तर्भाव "ggml.h"
#अन्तर्भाव "set.hpp"
#अन्तर्भाव <cstdint>
#अन्तर्भाव <sycl/sycl.hpp>
using नामाकाश sycl;

// Internal function: perform element-wise set operation for each thread
ओळीत शून्य set_f32(स्थिरांक दशांश* src, दशांश* dst,
                    स्थिरांक int64_t ne0, स्थिरांक int64_t ne1,
                    स्थिरांक int64_t ne2, स्थिरांक int64_t ne3,
                    स्थिरांक int64_t nb[3], स्थिरांक int64_t src_nb[3],
                    स्थिरांक int64_t offset_elem,
                    स्थिरांक nd_item<1>& item)
{
    स्थिरांक माप_प्रकार idx = item.get_global_id(0);
    स्थिरांक माप_प्रकार total = ne0 * ne1 * ne2 * ne3;
    जर (idx >= total) परत;

    // Convert linear index to 4D indices
    स्थिरांक माप_प्रकार i3 = idx / (ne2 * ne1 * ne0);
    स्थिरांक माप_प्रकार rem = idx % (ne2 * ne1 * ne0);
    स्थिरांक माप_प्रकार i2 = rem / (ne1 * ne0);
    स्थिरांक माप_प्रकार rem2 = rem % (ne1 * ne0);
    स्थिरांक माप_प्रकार i1 = rem2 / ne0;
    स्थिरांक माप_प्रकार i0 = rem2 % ne0;

    // Compute source and destination indices and copy
    dst[i0 + i1*nb[0] + i2*nb[1] + i3*nb[2] + offset_elem] =
        src[i0 + i1*src_nb[0] + i2*src_nb[1] + i3*src_nb[2]];
}

// Main function: prepare GPU queue and launch parallel_for
शून्य ggml_sycl_op_set(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    स्थिरांक ggml_tensor* src0 = dst->src[0];
    स्थिरांक ggml_tensor* src1 = dst->src[1];

    // Ensure shapes and types are compatible
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(dst->type == src0->type && src0->type == src1->type && dst->type == GGML_TYPE_F32);

    स्थिरांक int32_t* opts = (स्थिरांक int32_t*) dst->op_params;
    स्थिरांक int64_t nb[3]     = {opts[0]/आकार(दशांश), opts[1]/आकार(दशांश), opts[2]/आकार(दशांश)};
    स्थिरांक int64_t offset_elem = opts[3] / आकार(दशांश);
    स्थिरांक तार्किक inplace = opts[4];

    दशांश* dst_ptr = (दशांश*) dst->data;
    स्थिरांक दशांश* src0_ptr = (स्थिरांक दशांश*) src0->data;
    स्थिरांक दशांश* src1_ptr = (स्थिरांक दशांश*) src1->data;

    queue_ptr stream = ctx.stream();

    // Copy src0 to dst if not inplace
    जर (!inplace)
        stream->स_नकल(dst_ptr, src0_ptr, ggml_nbytes(dst));

    स्थिरांक int64_t ne[4] = {src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]};
    स्थिरांक int64_t src_nb[3] = {src1->nb[1]/आकार(दशांश), src1->nb[2]/आकार(दशांश), src1->nb[3]/आकार(दशांश)};

    स्थिरांक माप_प्रकार total_threads = ne[0]*ne[1]*ne[2]*ne[3];
    स्थिरांक माप_प्रकार grid_size = ((total_threads + SYCL_SET_BLOCK_SIZE - 1) / SYCL_SET_BLOCK_SIZE) * SYCL_SET_BLOCK_SIZE;

    // Copy src0 to dst if not inplace
    stream->parallel_for(
        nd_range<1>(range<1>(grid_size), range<1>(SYCL_SET_BLOCK_SIZE)),
        [=](nd_item<1> item) {
            set_f32(src1_ptr, dst_ptr,
                ne[0], ne[1], ne[2], ne[3],
                nb, src_nb, offset_elem, item); }
    );
}
