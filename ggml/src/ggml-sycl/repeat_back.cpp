#ضمن "repeat_back.hpp"

#ضمن "common.hpp"

#تعریف_کن GGML_ASSERT_TENSOR_میگنجد_INT(t) \
    GGML_ASSERT((t)->ne[0] < بیشترین_صحیح && (t)->ne[1] < بیشترین_صحیح && (t)->ne[2] < بیشترین_صحیح && (t)->ne[3] < بیشترین_صحیح)

فراغ ggml_sycl_op_repeat_پشت(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ثابت عائم * src0_dd = (ثابت عائم *) dst->src[0]->داده;
    عائم *       dst_dd  = (عائم *) dst->داده;

    GGML_ASSERT_TENSOR_میگنجد_INT(dst);
    GGML_ASSERT_TENSOR_میگنجد_INT(dst->src[0]);

    ثابت صحيح ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    ثابت صحيح ne00 = dst->src[0]->ne[0], ne01 = dst->src[0]->ne[1], ne02 = dst->src[0]->ne[2],
              ne03 = dst->src[0]->ne[3];

    ثابت صحيح nr0 = ne00 / ne0;
    ثابت صحيح nr1 = ne01 / ne1;
    ثابت صحيح nr2 = ne02 / ne2;
    ثابت صحيح nr3 = ne03 / ne3;

    ثابت صحيح nb0 = dst->src[0]->nb[0];
    ثابت صحيح nb1 = dst->src[0]->nb[1];
    ثابت صحيح nb2 = dst->src[0]->nb[2];
    ثابت صحيح nb3 = dst->src[0]->nb[3];

    ثابت حرف * پایه = (ثابت حرف *) src0_dd;

    ثابت size_t  مجموع      = (size_t) ne0 * ne1 * ne2 * ne3;
    constexpr صحيح بلوک_اندازه_2 = 256;
    ثابت صحيح     رقم_بلوکها = (مجموع + بلوک_اندازه_2 - 1) / بلوک_اندازه_2;

    ثابت عائم inv_ne0      = 1.0f / ne0;
    ثابت عائم inv_ne_01    = 1.0f / (ne0 * ne1);
    ثابت عائم inv_ne_012   = 1.0f / (ne0 * ne1 * ne2);
    ثابت صحيح   repeat_شمار = nr0 * nr1 * nr2 * nr3;

    صف_ptr stream = ctx.stream();

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(رقم_بلوکها * بلوک_اندازه_2), sycl::range<1>(بلوک_اندازه_2)),
        [=](sycl::nd_قلم<1> قلم_ct1) {
            ثابت size_t i = قلم_ct1.get_global_خطی_id();
            إذا (i >= مجموع) {
                أرجع;
            }

            ثابت صحيح i3 = (صحيح) (i * inv_ne_012);
            ثابت صحيح i2 = (صحيح) (i * inv_ne_01) - i3 * ne2;
            ثابت صحيح i1 = (صحيح) (i * inv_ne0) - (صحيح) (i * inv_ne_01) * ne1;
            ثابت صحيح i0 = i - (صحيح) (i * inv_ne0) * ne0;

            صحيح   j0 = 0, j1 = 0, j2 = 0, j3 = 0;
            عائم acc = 0.0f;

            لكل (صحيح j = 0; j < repeat_شمار; ++j) {
                ثابت عائم * ptr = (ثابت عائم *) (پایه + (i0 + j0 * ne0) * nb0 + (i1 + j1 * ne1) * nb1 +
                    (i2 + j2 * ne2) * nb2 + (i3 + j3 * ne3) * nb3);
                acc += *ptr;

                صحيح انتقال = (++j0 >= nr0);
                j0 -= انتقال * nr0;
                انتقال = (انتقال && (++j1 >= nr1));
                j1 -= انتقال * nr1;
                انتقال = (انتقال && (++j2 >= nr2));
                j2 -= انتقال * nr2;
                j3 += انتقال;
            }
            dst_dd[i] = acc;
        });
}
