#شامل "fuse_argsort_topk.h"

#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/slice.hpp>
#شامل <openvino/op/topk.hpp>
#شامل <openvino/pass/pattern/op/wrap_type.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام pass {

Fuse_Argsort_بالاترین_K::Fuse_Argsort_بالاترین_K() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    auto pattern = ov::pass::pattern::wrap_type<ov::op::v11::بالاترین_K>();

    ثابت auto callback = [](ov::pass::pattern::Matcher & m) {
        auto topk = ov::as_type_ptr<ov::op::v11::بالاترین_K>(m.get_match_ریشه());
        اگر (!topk->output(0).get_target_inputs().empty() || topk->output(1).get_target_inputs().empty() ||
            topk->get_sort_type() != ov::op::v11::بالاترین_K::SortType::SORT_مقادیر) {
            بازگشت نادرست;
        }

        ثابت auto shape = topk->get_output_partial_shape(1);
        اگر (shape.rank().is_dynamic()) {
            بازگشت نادرست;
        }
        ثابت int64_t rank = shape.rank().get_length();
        ثابت int64_t axis = topk->get_axis();
        اگر (shape[axis].is_dynamic()) {
            بازگشت نادرست;
        }

        int64_t prefix = -1;
        برای (ثابت auto & درون_داد : topk->output(1).get_target_inputs()) {
            ثابت auto * slice = ov::as_type<ov::op::v8::Slice>(درون_داد.get_گره());
            اگر (!slice || درون_داد.get_index() != 0 || slice->get_درون_داد_size() != 5) {
                بازگشت نادرست;
            }
            int64_t مقادیر[4];
            برای (size_t i = 0; i < 4; ++i) {
                auto value = ov::as_type_ptr<ov::op::v0::Constant>(slice->get_درون_داد_گره_shared_ptr(i + 1));
                اگر (!value || ov::shape_size(value->get_shape()) != 1) {
                    بازگشت نادرست;
                }
                مقادیر[i] = value->cast_vector<int64_t>()[0];
            }
            ثابت int64_t slice_axis = مقادیر[3] < 0 ? مقادیر[3] + rank : مقادیر[3];
            اگر (مقادیر[0] != 0 || مقادیر[2] != 1 || slice_axis != axis || مقادیر[1] <= 0 ||
                مقادیر[1] >= shape[axis].get_length() || (prefix != -1 && prefix != مقادیر[1])) {
                بازگشت نادرست;
            }
            prefix = مقادیر[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->set_argument(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        بازگشت درست;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
