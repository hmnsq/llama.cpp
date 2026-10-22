#ضمن "repeat_back.hpp"

#ضمن "common.hpp"

#وصف_ڪريو GGML_ASSERT_TENSOR_سمائي_INT(t) \
    GGML_ASSERT((t)->ne[0] < وڌ_کان_وڌ_عدد && (t)->ne[1] < وڌ_کان_وڌ_عدد && (t)->ne[2] < وڌ_کان_وڌ_عدد && (t)->ne[3] < وڌ_کان_وڌ_عدد)

فراغ ggml_sycl_op_repeat_پٺ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ثابت عائم * src0_dd = (ثابت عائم *) dst->src[0]->معلومات;
    عائم *       dst_dd  = (عائم *) dst->معلومات;

    GGML_ASSERT_TENSOR_سمائي_INT(dst);
    GGML_ASSERT_TENSOR_سمائي_INT(dst->src[0]);

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

    ثابت حرف * بنياد = (ثابت حرف *) src0_dd;

    ثابت size_t  ڪل      = (size_t) ne0 * ne1 * ne2 * ne3;
    constexpr صحيح بلاڪ_سائيز_2 = 256;
    ثابت صحيح     انگ_بلاڪون = (ڪل + بلاڪ_سائيز_2 - 1) / بلاڪ_سائيز_2;

    ثابت عائم inv_ne0      = 1.0f / ne0;
    ثابت عائم inv_ne_01    = 1.0f / (ne0 * ne1);
    ثابت عائم inv_ne_012   = 1.0f / (ne0 * ne1 * ne2);
    ثابت صحيح   repeat_ڳڻپ = nr0 * nr1 * nr2 * nr3;

    قطار_ptr stream = ctx.stream();

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(انگ_بلاڪون * بلاڪ_سائيز_2), sycl::range<1>(بلاڪ_سائيز_2)),
        [=](sycl::nd_شي<1> شي_ct1) {
            ثابت size_t i = شي_ct1.get_global_لڪيري_id();
            إذا (i >= ڪل) {
                أرجع;
            }

            ثابت صحيح i3 = (صحيح) (i * inv_ne_012);
            ثابت صحيح i2 = (صحيح) (i * inv_ne_01) - i3 * ne2;
            ثابت صحيح i1 = (صحيح) (i * inv_ne0) - (صحيح) (i * inv_ne_01) * ne1;
            ثابت صحيح i0 = i - (صحيح) (i * inv_ne0) * ne0;

            صحيح   j0 = 0, j1 = 0, j2 = 0, j3 = 0;
            عائم acc = 0.0f;

            لكل (صحيح j = 0; j < repeat_ڳڻپ; ++j) {
                ثابت عائم * ptr = (ثابت عائم *) (بنياد + (i0 + j0 * ne0) * nb0 + (i1 + j1 * ne1) * nb1 +
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
