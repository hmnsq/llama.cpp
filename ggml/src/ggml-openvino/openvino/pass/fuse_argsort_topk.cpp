#ضمن "fuse_argsort_topk.h"

#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/topk.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

Fuse_Argsort_بالاترین_K::Fuse_Argsort_بالاترین_K() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    خودکار pattern = ov::pass::pattern::wrap_type<ov::op::v11::بالاترین_K>();

    ثابت خودکار callback = [](ov::pass::pattern::Matcher & m) {
        خودکار topk = ov::as_type_ptr<ov::op::v11::بالاترین_K>(m.get_تطابق_ریشه());
        إذا (!topk->output(0).get_هدف_inputs().خالی() || topk->output(1).get_هدف_inputs().خالی() ||
            topk->get_مرتب_سازی_type() != ov::op::v11::بالاترین_K::مرتب_سازی_Type::مرتب_سازی_مقادیر) {
            أرجع خطأ;
        }

        ثابت خودکار shape = topk->get_output_partial_shape(1);
        إذا (shape.rank().is_dynamic()) {
            أرجع خطأ;
        }
        ثابت int64_t rank = shape.rank().get_درازی();
        ثابت int64_t axis = topk->get_axis();
        إذا (shape[axis].is_dynamic()) {
            أرجع خطأ;
        }

        int64_t prefix = -1;
        لكل (ثابت خودکار & درون_داد : topk->output(1).get_هدف_inputs()) {
            ثابت خودکار * برش_2 = ov::as_type<ov::op::v8::برش>(درون_داد.get_گره());
            إذا (!برش_2 || درون_داد.get_شاخص() != 0 || برش_2->get_درون_داد_اندازه() != 5) {
                أرجع خطأ;
            }
            int64_t مقادیر[4];
            لكل (size_t i = 0; i < 4; ++i) {
                خودکار value = ov::as_type_ptr<ov::op::v0::Constant>(برش_2->get_درون_داد_گره_shared_ptr(i + 1));
                إذا (!value || ov::shape_اندازه(value->get_shape()) != 1) {
                    أرجع خطأ;
                }
                مقادیر[i] = value->cast_وکتور<int64_t>()[0];
            }
            ثابت int64_t برش_axis = مقادیر[3] < 0 ? مقادیر[3] + rank : مقادیر[3];
            إذا (مقادیر[0] != 0 || مقادیر[2] != 1 || برش_axis != axis || مقادیر[1] <= 0 ||
                مقادیر[1] >= shape[axis].get_درازی() || (prefix != -1 && prefix != مقادیر[1])) {
                أرجع خطأ;
            }
            prefix = مقادیر[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->مجموعه_آرگومان(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        أرجع درست;
    };

    ثبات_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
