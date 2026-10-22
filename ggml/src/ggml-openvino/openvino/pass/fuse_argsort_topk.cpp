#ضمن "fuse_argsort_topk.h"

#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/topk.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

Fuse_Argsort_چوٽي_K::Fuse_Argsort_چوٽي_K() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    خودڪار pattern = ov::pass::pattern::wrap_type<ov::op::v11::چوٽي_K>();

    ثابت خودڪار callback = [](ov::pass::pattern::Matcher & m) {
        خودڪار topk = ov::as_type_ptr<ov::op::v11::چوٽي_K>(m.get_ميلاپ_پاڙ());
        إذا (!topk->output(0).get_هدف_inputs().خالي_جايون() || topk->output(1).get_هدف_inputs().خالي_جايون() ||
            topk->get_ترتيب_ڏيو_type() != ov::op::v11::چوٽي_K::ترتيب_ڏيو_Type::ترتيب_ڏيو_قدرون) {
            أرجع خطأ;
        }

        ثابت خودڪار shape = topk->get_output_partial_shape(1);
        إذا (shape.rank().is_dynamic()) {
            أرجع خطأ;
        }
        ثابت int64_t rank = shape.rank().get_ڊيگهائي();
        ثابت int64_t axis = topk->get_axis();
        إذا (shape[axis].is_dynamic()) {
            أرجع خطأ;
        }

        int64_t prefix = -1;
        لكل (ثابت خودڪار & داخلا_2 : topk->output(1).get_هدف_inputs()) {
            ثابت خودڪار * ٽڪرو_2 = ov::as_type<ov::op::v8::ٽڪرو>(داخلا_2.get_ڳنڍ());
            إذا (!ٽڪرو_2 || داخلا_2.get_اشاريو() != 0 || ٽڪرو_2->get_داخلا_سائيز() != 5) {
                أرجع خطأ;
            }
            int64_t قدرون[4];
            لكل (size_t i = 0; i < 4; ++i) {
                خودڪار value = ov::as_type_ptr<ov::op::v0::Constant>(ٽڪرو_2->get_داخلا_ڳنڍ_shared_ptr(i + 1));
                إذا (!value || ov::shape_سائيز(value->get_shape()) != 1) {
                    أرجع خطأ;
                }
                قدرون[i] = value->cast_ويڪٽر<int64_t>()[0];
            }
            ثابت int64_t ٽڪرو_axis = قدرون[3] < 0 ? قدرون[3] + rank : قدرون[3];
            إذا (قدرون[0] != 0 || قدرون[2] != 1 || ٽڪرو_axis != axis || قدرون[1] <= 0 ||
                قدرون[1] >= shape[axis].get_ڊيگهائي() || (prefix != -1 && prefix != قدرون[1])) {
                أرجع خطأ;
            }
            prefix = قدرون[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->سيٽ_دليل(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        أرجع سچ;
    };

    رجسٽر_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
