#অন্তর্ভাৱ "fuse_argsort_topk.h"

#অন্তর্ভাৱ <openvino/op/constant.hpp>
#অন্তর্ভাৱ <openvino/op/slice.hpp>
#অন্তর্ভাৱ <openvino/op/topk.hpp>
#অন্তর্ভাৱ <openvino/pass/pattern/op/wrap_type.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ pass {

Fuse_Argsort_শীৰ্ষ_K::Fuse_Argsort_শীৰ্ষ_K() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    auto pattern = ov::pass::pattern::wrap_type<ov::op::v11::শীৰ্ষ_K>();

    ধ্রুৱ auto callback = [](ov::pass::pattern::Matcher & m) {
        auto topk = ov::as_type_ptr<ov::op::v11::শীৰ্ষ_K>(m.get_match_মূল());
        যদি (!topk->output(0).get_target_inputs().empty() || topk->output(1).get_target_inputs().empty() ||
            topk->get_sort_type() != ov::op::v11::শীৰ্ষ_K::SortType::SORT_মানসমূহ) {
            ঘূৰাই মিছা;
        }

        ধ্রুৱ auto shape = topk->get_output_partial_shape(1);
        যদি (shape.rank().is_dynamic()) {
            ঘূৰাই মিছা;
        }
        ধ্রুৱ int64_t rank = shape.rank().get_length();
        ধ্রুৱ int64_t axis = topk->get_axis();
        যদি (shape[axis].is_dynamic()) {
            ঘূৰাই মিছা;
        }

        int64_t prefix = -1;
        বাবে (ধ্রুৱ auto & প্ৰবেশ : topk->output(1).get_target_inputs()) {
            ধ্রুৱ auto * slice = ov::as_type<ov::op::v8::Slice>(প্ৰবেশ.get_গ্ৰন্থি());
            যদি (!slice || প্ৰবেশ.get_index() != 0 || slice->get_প্ৰবেশ_size() != 5) {
                ঘূৰাই মিছা;
            }
            int64_t মানসমূহ[4];
            বাবে (size_t i = 0; i < 4; ++i) {
                auto value = ov::as_type_ptr<ov::op::v0::Constant>(slice->get_প্ৰবেশ_গ্ৰন্থি_shared_ptr(i + 1));
                যদি (!value || ov::shape_size(value->get_shape()) != 1) {
                    ঘূৰাই মিছা;
                }
                মানসমূহ[i] = value->cast_vector<int64_t>()[0];
            }
            ধ্রুৱ int64_t slice_axis = মানসমূহ[3] < 0 ? মানসমূহ[3] + rank : মানসমূহ[3];
            যদি (মানসমূহ[0] != 0 || মানসমূহ[2] != 1 || slice_axis != axis || মানসমূহ[1] <= 0 ||
                মানসমূহ[1] >= shape[axis].get_length() || (prefix != -1 && prefix != মানসমূহ[1])) {
                ঘূৰাই মিছা;
            }
            prefix = মানসমূহ[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->set_argument(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        ঘূৰাই সঁচা;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
