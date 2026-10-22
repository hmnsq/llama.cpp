#شامل "repeat_back.hpp"

#شامل "common.hpp"

#تعریف_کرو GGML_ASSERT_TENSOR_سماتا_INT(t) \
    GGML_ASSERT((t)->ne[0] < زیادہ_سے_زیادہ_عدد && (t)->ne[1] < زیادہ_سے_زیادہ_عدد && (t)->ne[2] < زیادہ_سے_زیادہ_عدد && (t)->ne[3] < زیادہ_سے_زیادہ_عدد)

خالی ggml_sycl_op_repeat_عقب(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ثابت اعشاری * src0_dd = (ثابت اعشاری *) dst->src[0]->ڈیٹا;
    اعشاری *       dst_dd  = (اعشاری *) dst->ڈیٹا;

    GGML_ASSERT_TENSOR_سماتا_INT(dst);
    GGML_ASSERT_TENSOR_سماتا_INT(dst->src[0]);

    ثابت صحیح ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    ثابت صحیح ne00 = dst->src[0]->ne[0], ne01 = dst->src[0]->ne[1], ne02 = dst->src[0]->ne[2],
              ne03 = dst->src[0]->ne[3];

    ثابت صحیح nr0 = ne00 / ne0;
    ثابت صحیح nr1 = ne01 / ne1;
    ثابت صحیح nr2 = ne02 / ne2;
    ثابت صحیح nr3 = ne03 / ne3;

    ثابت صحیح nb0 = dst->src[0]->nb[0];
    ثابت صحیح nb1 = dst->src[0]->nb[1];
    ثابت صحیح nb2 = dst->src[0]->nb[2];
    ثابت صحیح nb3 = dst->src[0]->nb[3];

    ثابت حرف * بنیاد = (ثابت حرف *) src0_dd;

    ثابت size_t  کل      = (size_t) ne0 * ne1 * ne2 * ne3;
    constexpr صحیح بلاک_سائز_2 = 256;
    ثابت صحیح     عدد_بلاکیں = (کل + بلاک_سائز_2 - 1) / بلاک_سائز_2;

    ثابت اعشاری inv_ne0      = 1.0f / ne0;
    ثابت اعشاری inv_ne_01    = 1.0f / (ne0 * ne1);
    ثابت اعشاری inv_ne_012   = 1.0f / (ne0 * ne1 * ne2);
    ثابت صحیح   repeat_گنتی = nr0 * nr1 * nr2 * nr3;

    قطار_ptr stream = ctx.stream();

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(عدد_بلاکیں * بلاک_سائز_2), sycl::range<1>(بلاک_سائز_2)),
        [=](sycl::nd_شے<1> شے_ct1) {
            ثابت size_t i = شے_ct1.get_global_خطی_id();
            اگر (i >= کل) {
                واپس;
            }

            ثابت صحیح i3 = (صحیح) (i * inv_ne_012);
            ثابت صحیح i2 = (صحیح) (i * inv_ne_01) - i3 * ne2;
            ثابت صحیح i1 = (صحیح) (i * inv_ne0) - (صحیح) (i * inv_ne_01) * ne1;
            ثابت صحیح i0 = i - (صحیح) (i * inv_ne0) * ne0;

            صحیح   j0 = 0, j1 = 0, j2 = 0, j3 = 0;
            اعشاری acc = 0.0f;

            برائے (صحیح j = 0; j < repeat_گنتی; ++j) {
                ثابت اعشاری * ptr = (ثابت اعشاری *) (بنیاد + (i0 + j0 * ne0) * nb0 + (i1 + j1 * ne1) * nb1 +
                    (i2 + j2 * ne2) * nb2 + (i3 + j3 * ne3) * nb3);
                acc += *ptr;

                صحیح حاصل = (++j0 >= nr0);
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
