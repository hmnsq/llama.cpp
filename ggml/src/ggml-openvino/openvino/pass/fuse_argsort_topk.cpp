#अन्तर्भाव "fuse_argsort_topk.h"

#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/topk.hpp>
#अन्तर्भाव <openvino/pass/pattern/op/wrap_type.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश pass {

Fuse_Argsort_शिखरम्_K::Fuse_Argsort_शिखरम्_K() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    auto pattern = ov::pass::pattern::wrap_type<ov::op::v11::शिखरम्_K>();

    ध्रुव auto callback = [](ov::pass::pattern::Matcher & m) {
        auto topk = ov::as_type_ptr<ov::op::v11::शिखरम्_K>(m.get_match_मूलम्());
        सचे (!topk->output(0).get_target_inputs().empty() || topk->output(1).get_target_inputs().empty() ||
            topk->get_sort_type() != ov::op::v11::शिखरम्_K::SortType::SORT_मानानि) {
            पटिनिवत्त मिच्छा;
        }

        ध्रुव auto shape = topk->get_output_partial_shape(1);
        सचे (shape.rank().is_dynamic()) {
            पटिनिवत्त मिच्छा;
        }
        ध्रुव int64_t rank = shape.rank().get_length();
        ध्रुव int64_t axis = topk->get_axis();
        सचे (shape[axis].is_dynamic()) {
            पटिनिवत्त मिच्छा;
        }

        int64_t prefix = -1;
        अत्थाय (ध्रुव auto & निवेशः : topk->output(1).get_target_inputs()) {
            ध्रुव auto * slice = ov::as_type<ov::op::v8::Slice>(निवेशः.get_ग्रन्थिः());
            सचे (!slice || निवेशः.get_index() != 0 || slice->get_निवेशः_size() != 5) {
                पटिनिवत्त मिच्छा;
            }
            int64_t मानानि[4];
            अत्थाय (size_t i = 0; i < 4; ++i) {
                auto value = ov::as_type_ptr<ov::op::v0::Constant>(slice->get_निवेशः_ग्रन्थिः_shared_ptr(i + 1));
                सचे (!value || ov::shape_size(value->get_shape()) != 1) {
                    पटिनिवत्त मिच्छा;
                }
                मानानि[i] = value->cast_vector<int64_t>()[0];
            }
            ध्रुव int64_t slice_axis = मानानि[3] < 0 ? मानानि[3] + rank : मानानि[3];
            सचे (मानानि[0] != 0 || मानानि[2] != 1 || slice_axis != axis || मानानि[1] <= 0 ||
                मानानि[1] >= shape[axis].get_length() || (prefix != -1 && prefix != मानानि[1])) {
                पटिनिवत्त मिच्छा;
            }
            prefix = मानानि[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->set_argument(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        पटिनिवत्त सच्च;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
