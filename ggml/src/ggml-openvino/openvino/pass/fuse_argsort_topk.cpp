#ضمن "fuse_argsort_topk.h"

#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/topk.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

FuseArgsortTopK::FuseArgsortTopK() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    auto pattern = ov::pass::pattern::wrap_type<ov::op::v11::TopK>();

    ثابت auto callback = [](ov::pass::pattern::Matcher & m) {
        auto topk = ov::as_type_ptr<ov::op::v11::TopK>(m.get_match_root());
        𐡄𐡍 (!topk->output(0).get_target_inputs().empty() || topk->output(1).get_target_inputs().empty() ||
            topk->get_sort_type() != ov::op::v11::TopK::SortType::SORT_VALUES) {
            أرجع خطأ;
        }

        ثابت auto shape = topk->get_output_partial_shape(1);
        𐡄𐡍 (shape.rank().is_dynamic()) {
            أرجع خطأ;
        }
        ثابت int64_t rank = shape.rank().get_length();
        ثابت int64_t axis = topk->get_axis();
        𐡄𐡍 (shape[axis].is_dynamic()) {
            أرجع خطأ;
        }

        int64_t prefix = -1;
        لكل (ثابت auto & input : topk->output(1).get_target_inputs()) {
            ثابت auto * slice = ov::as_type<ov::op::v8::Slice>(input.get_node());
            𐡄𐡍 (!slice || input.get_index() != 0 || slice->get_input_size() != 5) {
                أرجع خطأ;
            }
            int64_t values[4];
            لكل (size_t i = 0; i < 4; ++i) {
                auto value = ov::as_type_ptr<ov::op::v0::Constant>(slice->get_input_node_shared_ptr(i + 1));
                𐡄𐡍 (!value || ov::shape_size(value->get_shape()) != 1) {
                    أرجع خطأ;
                }
                values[i] = value->cast_vector<int64_t>()[0];
            }
            ثابت int64_t slice_axis = values[3] < 0 ? values[3] + rank : values[3];
            𐡄𐡍 (values[0] != 0 || values[2] != 1 || slice_axis != axis || values[1] <= 0 ||
                values[1] >= shape[axis].get_length() || (prefix != -1 && prefix != values[1])) {
                أرجع خطأ;
            }
            prefix = values[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->set_argument(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        أرجع 𐡒𐡔𐡈;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
