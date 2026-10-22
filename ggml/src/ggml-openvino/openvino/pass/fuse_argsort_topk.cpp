#ضمن "fuse_argsort_topk.h"

#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/topk.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

Fuse_Argsort_چوٹی_K::Fuse_Argsort_چوٹی_K() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    خودکار pattern = ov::pass::pattern::wrap_type<ov::op::v11::چوٹی_K>();

    ثابت خودکار callback = [](ov::pass::pattern::Matcher & m) {
        خودکار topk = ov::as_type_ptr<ov::op::v11::چوٹی_K>(m.get_میل_جڑ());
        إذا (!topk->output(0).get_نشانہ_inputs().خالی_تھاواں() || topk->output(1).get_نشانہ_inputs().خالی_تھاواں() ||
            topk->get_ترتیب_دیو_type() != ov::op::v11::چوٹی_K::ترتیب_دیو_Type::ترتیب_دیو_اقدار) {
            أرجع خطأ;
        }

        ثابت خودکار shape = topk->get_output_partial_shape(1);
        إذا (shape.rank().is_dynamic()) {
            أرجع خطأ;
        }
        ثابت int64_t rank = shape.rank().get_لمائی();
        ثابت int64_t axis = topk->get_axis();
        إذا (shape[axis].is_dynamic()) {
            أرجع خطأ;
        }

        int64_t prefix = -1;
        لكل (ثابت خودکار & داخلہ : topk->output(1).get_نشانہ_inputs()) {
            ثابت خودکار * ٹکڑا_2 = ov::as_type<ov::op::v8::ٹکڑا>(داخلہ.get_گرہ());
            إذا (!ٹکڑا_2 || داخلہ.get_اشاریہ() != 0 || ٹکڑا_2->get_داخلہ_ناپ() != 5) {
                أرجع خطأ;
            }
            int64_t اقدار[4];
            لكل (size_t i = 0; i < 4; ++i) {
                خودکار value = ov::as_type_ptr<ov::op::v0::Constant>(ٹکڑا_2->get_داخلہ_گرہ_shared_ptr(i + 1));
                إذا (!value || ov::shape_ناپ(value->get_shape()) != 1) {
                    أرجع خطأ;
                }
                اقدار[i] = value->cast_سمتیہ<int64_t>()[0];
            }
            ثابت int64_t ٹکڑا_axis = اقدار[3] < 0 ? اقدار[3] + rank : اقدار[3];
            إذا (اقدار[0] != 0 || اقدار[2] != 1 || ٹکڑا_axis != axis || اقدار[1] <= 0 ||
                اقدار[1] >= shape[axis].get_لمائی() || (prefix != -1 && prefix != اقدار[1])) {
                أرجع خطأ;
            }
            prefix = اقدار[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->سیٹ_دلیل(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        أرجع سچ;
    };

    رجسٹر_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
