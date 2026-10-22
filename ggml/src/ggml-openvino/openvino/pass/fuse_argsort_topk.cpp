#ضمن "fuse_argsort_topk.h"

#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/topk.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

Fuse_Argsort_سر_K::Fuse_Argsort_سر_K() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    خپلکاری pattern = ov::pass::pattern::wrap_type<ov::op::v11::سر_K>();

    ثابت خپلکاری callback = [](ov::pass::pattern::Matcher & m) {
        خپلکاری topk = ov::as_type_ptr<ov::op::v11::سر_K>(m.get_سمون_ریښه());
        إذا (!topk->output(0).get_موخه_inputs().تشه() || topk->output(1).get_موخه_inputs().تشه() ||
            topk->get_ترتیبول_type() != ov::op::v11::سر_K::ترتیبول_Type::ترتیبول_ارزښتونه) {
            أرجع خطأ;
        }

        ثابت خپلکاری shape = topk->get_output_partial_shape(1);
        إذا (shape.rank().is_dynamic()) {
            أرجع خطأ;
        }
        ثابت int64_t rank = shape.rank().get_اوږدتیا();
        ثابت int64_t axis = topk->get_axis();
        إذا (shape[axis].is_dynamic()) {
            أرجع خطأ;
        }

        int64_t prefix = -1;
        لكل (ثابت خپلکاری & ننوت : topk->output(1).get_موخه_inputs()) {
            ثابت خپلکاری * ټوټه_2 = ov::as_type<ov::op::v8::ټوټه>(ننوت.get_غوټه());
            إذا (!ټوټه_2 || ننوت.get_شاخص() != 0 || ټوټه_2->get_ننوت_اندازه() != 5) {
                أرجع خطأ;
            }
            int64_t ارزښتونه[4];
            لكل (size_t i = 0; i < 4; ++i) {
                خپلکاری value = ov::as_type_ptr<ov::op::v0::Constant>(ټوټه_2->get_ننوت_غوټه_shared_ptr(i + 1));
                إذا (!value || ov::shape_اندازه(value->get_shape()) != 1) {
                    أرجع خطأ;
                }
                ارزښتونه[i] = value->cast_ویکټور<int64_t>()[0];
            }
            ثابت int64_t ټوټه_axis = ارزښتونه[3] < 0 ? ارزښتونه[3] + rank : ارزښتونه[3];
            إذا (ارزښتونه[0] != 0 || ارزښتونه[2] != 1 || ټوټه_axis != axis || ارزښتونه[1] <= 0 ||
                ارزښتونه[1] >= shape[axis].get_اوږدتیا() || (prefix != -1 && prefix != ارزښتونه[1])) {
                أرجع خطأ;
            }
            prefix = ارزښتونه[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->سېټ_دلیل(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        أرجع ریښتیا;
    };

    ثبتوونکی_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
