#ضمن "fuse_argsort_topk.h"

#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/topk.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

Fuse_Argsort_اتس_K::Fuse_Argsort_اتس_K() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    auto pattern = ov::pass::pattern::wrap_type<ov::op::v11::اتس_K>();

    ثابت auto callback = [](ov::pass::pattern::Matcher & m) {
        auto topk = ov::as_type_ptr<ov::op::v11::اتس_K>(m.get_match_اکر());
        إذا (!topk->output(0).get_target_inputs().empty() || topk->output(1).get_target_inputs().empty() ||
            topk->get_sort_type() != ov::op::v11::اتس_K::SortType::SORT_نيلاي_سموا) {
            أرجع خطأ;
        }

        ثابت auto shape = topk->get_output_partial_shape(1);
        إذا (shape.rank().is_dynamic()) {
            أرجع خطأ;
        }
        ثابت int64_t rank = shape.rank().get_length();
        ثابت int64_t axis = topk->get_axis();
        إذا (shape[axis].is_dynamic()) {
            أرجع خطأ;
        }

        int64_t prefix = -1;
        لكل (ثابت auto & ماسوقن : topk->output(1).get_target_inputs()) {
            ثابت auto * slice = ov::as_type<ov::op::v8::Slice>(ماسوقن.get_نود());
            إذا (!slice || ماسوقن.get_index() != 0 || slice->get_ماسوقن_size() != 5) {
                أرجع خطأ;
            }
            int64_t نيلاي_سموا[4];
            لكل (size_t i = 0; i < 4; ++i) {
                auto value = ov::as_type_ptr<ov::op::v0::Constant>(slice->get_ماسوقن_نود_shared_ptr(i + 1));
                إذا (!value || ov::shape_size(value->get_shape()) != 1) {
                    أرجع خطأ;
                }
                نيلاي_سموا[i] = value->cast_vector<int64_t>()[0];
            }
            ثابت int64_t slice_axis = نيلاي_سموا[3] < 0 ? نيلاي_سموا[3] + rank : نيلاي_سموا[3];
            إذا (نيلاي_سموا[0] != 0 || نيلاي_سموا[2] != 1 || slice_axis != axis || نيلاي_سموا[1] <= 0 ||
                نيلاي_سموا[1] >= shape[axis].get_length() || (prefix != -1 && prefix != نيلاي_سموا[1])) {
                أرجع خطأ;
            }
            prefix = نيلاي_سموا[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->set_argument(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        أرجع true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
