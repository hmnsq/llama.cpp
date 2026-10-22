#ضمن "diag.hpp"
#ضمن "common.hpp"

#تعریف_کن SYCL_DIAG_بلوک_اندازه 256

قالب <نام_نوع T>
ساكن فراغ diag_kernel(T * __restrict__ dst, ثابت T * __restrict__ src,
                        ثابت int64_t ne0, ثابت int64_t ne1,
                        ثابت int64_t ne2, ثابت int64_t ne3,
                        ثابت int64_t مجموع_elements,
                        ثابت sycl::nd_قلم<1> & قلم) {
    ثابت int64_t i = قلم.get_global_id(0);
    إذا (i >= مجموع_elements) {
        أرجع;
    }

    ثابت int64_t i0 = i % ne0;
    ثابت int64_t i1 = (i / ne0) % ne1;
    ثابت int64_t i2 = (i / (ne0 * ne1)) % ne2;
    ثابت int64_t i3 = i / (ne0 * ne1 * ne2);

    ثابت int64_t dst_idx = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;

    إذا (i0 == i1) {
        ثابت int64_t batch_idx = i3 * ne2 + i2;
        dst[dst_idx] = src[batch_idx * ne0 + i0];
    } وإلا {
        dst[dst_idx] = T(0);
    }

    (فراغ)ne3;
}

مضمن فراغ ggml_sycl_op_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->ne[1] == 1);

    dpct::صف_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_مجموعه_device(ctx.device));

    ثابت فراغ * src0_d = src0->داده;
    فراغ * dst_d = dst->داده;

    ثابت int64_t ne0 = dst->ne[0];
    ثابت int64_t ne1 = dst->ne[1];
    ثابت int64_t ne2 = dst->ne[2];
    ثابت int64_t ne3 = dst->ne[3];
    ثابت int64_t n_elems = ggml_nelements(dst);
    ثابت int64_t رقم_بلوکها = (n_elems + SYCL_DIAG_بلوک_اندازه - 1) / SYCL_DIAG_بلوک_اندازه;

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    stream->parallel_for(
        sycl::nd_range<1>(رقم_بلوکها * SYCL_DIAG_بلوک_اندازه, SYCL_DIAG_بلوک_اندازه),
        [=](sycl::nd_قلم<1> قلم) {
            diag_kernel(static_cast<عائم *>(dst_d),
                        static_cast<ثابت عائم *>(src0_d),
                        ne0, ne1, ne2, ne3, n_elems, قلم);
        });
}

فراغ ggml_sycl_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag(ctx, dst);
}
