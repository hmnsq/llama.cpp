#समावेश "diag.hpp"
#समावेश "common.hpp"

#घोषणा SYCL_DIAG_BLOCK_आकार 256

ढाँचा <typename T>
अटल व्योम diag_kernel(T * __restrict__ dst, स्थिर T * __restrict__ src,
                        स्थिर int64_t ne0, स्थिर int64_t ne1,
                        स्थिर int64_t ne2, स्थिर int64_t ne3,
                        स्थिर int64_t योग_elements,
                        स्थिर sycl::nd_item<1> & item) {
    स्थिर int64_t i = item.get_global_id(0);
    अगर (i >= योग_elements) {
        वापस;
    }

    स्थिर int64_t i0 = i % ne0;
    स्थिर int64_t i1 = (i / ne0) % ne1;
    स्थिर int64_t i2 = (i / (ne0 * ne1)) % ne2;
    स्थिर int64_t i3 = i / (ne0 * ne1 * ne2);

    स्थिर int64_t dst_idx = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;

    अगर (i0 == i1) {
        स्थिर int64_t batch_idx = i3 * ne2 + i2;
        dst[dst_idx] = src[batch_idx * ne0 + i0];
    } अन्यथा {
        dst[dst_idx] = T(0);
    }

    (व्योम)ne3;
}

अंतरभूत व्योम ggml_sycl_op_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->ne[1] == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    स्थिर व्योम * src0_d = src0->तथ्य;
    व्योम * dst_d = dst->तथ्य;

    स्थिर int64_t ne0 = dst->ne[0];
    स्थिर int64_t ne1 = dst->ne[1];
    स्थिर int64_t ne2 = dst->ne[2];
    स्थिर int64_t ne3 = dst->ne[3];
    स्थिर int64_t n_elems = ggml_nelements(dst);
    स्थिर int64_t अंक_blocks = (n_elems + SYCL_DIAG_BLOCK_आकार - 1) / SYCL_DIAG_BLOCK_आकार;

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    stream->parallel_for(
        sycl::nd_range<1>(अंक_blocks * SYCL_DIAG_BLOCK_आकार, SYCL_DIAG_BLOCK_आकार),
        [=](sycl::nd_item<1> item) {
            diag_kernel(static_cast<भग्न *>(dst_d),
                        static_cast<स्थिर भग्न *>(src0_d),
                        ne0, ne1, ne2, ne3, n_elems, item);
        });
}

व्योम ggml_sycl_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag(ctx, dst);
}
