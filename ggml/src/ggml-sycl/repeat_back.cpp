#ضمن "repeat_back.hpp"

#ضمن "common.hpp"

#تعریف_کرو GGML_ASSERT_TENSOR_سماندا_INT(t) \
    GGML_ASSERT((t)->ne[0] < ودھ_توں_ودھ_عدد && (t)->ne[1] < ودھ_توں_ودھ_عدد && (t)->ne[2] < ودھ_توں_ودھ_عدد && (t)->ne[3] < ودھ_توں_ودھ_عدد)

فراغ ggml_sycl_op_repeat_پٹھ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ثابت عائم * src0_dd = (ثابت عائم *) dst->src[0]->انکڑے;
    عائم *       dst_dd  = (عائم *) dst->انکڑے;

    GGML_ASSERT_TENSOR_سماندا_INT(dst);
    GGML_ASSERT_TENSOR_سماندا_INT(dst->src[0]);

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

    ثابت حرف * ادھار = (ثابت حرف *) src0_dd;

    ثابت size_t  کل      = (size_t) ne0 * ne1 * ne2 * ne3;
    constexpr صحيح بلاک_ناپ_2 = 256;
    ثابت صحيح     عدد_بلاکاں = (کل + بلاک_ناپ_2 - 1) / بلاک_ناپ_2;

    ثابت عائم inv_ne0      = 1.0f / ne0;
    ثابت عائم inv_ne_01    = 1.0f / (ne0 * ne1);
    ثابت عائم inv_ne_012   = 1.0f / (ne0 * ne1 * ne2);
    ثابت صحيح   repeat_گنتی = nr0 * nr1 * nr2 * nr3;

    قطار_ptr stream = ctx.stream();

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(عدد_بلاکاں * بلاک_ناپ_2), sycl::range<1>(بلاک_ناپ_2)),
        [=](sycl::nd_شے<1> شے_ct1) {
            ثابت size_t i = شے_ct1.get_global_خطی_id();
            إذا (i >= کل) {
                أرجع;
            }

            ثابت صحيح i3 = (صحيح) (i * inv_ne_012);
            ثابت صحيح i2 = (صحيح) (i * inv_ne_01) - i3 * ne2;
            ثابت صحيح i1 = (صحيح) (i * inv_ne0) - (صحيح) (i * inv_ne_01) * ne1;
            ثابت صحيح i0 = i - (صحيح) (i * inv_ne0) * ne0;

            صحيح   j0 = 0, j1 = 0, j2 = 0, j3 = 0;
            عائم acc = 0.0f;

            لكل (صحيح j = 0; j < repeat_گنتی; ++j) {
                ثابت عائم * ptr = (ثابت عائم *) (ادھار + (i0 + j0 * ne0) * nb0 + (i1 + j1 * ne1) * nb1 +
                    (i2 + j2 * ne2) * nb2 + (i3 + j3 * ne3) * nb3);
                acc += *ptr;

                صحيح حاصل = (++j0 >= nr0);
                j0 -= حاصل * nr0;
                حاصل = (حاصل && (++j1 >= nr1));
                j1 -= حاصل * nr1;
                حاصل = (حاصل && (++j2 >= nr2));
                j2 -= حاصل * nr2;
                j3 += حاصل;
            }
            dst_dd[i] = acc;
        });
}
