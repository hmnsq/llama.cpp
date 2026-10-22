#অন্তর্ভাব "fuse_argsort_topk.h"

#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/slice.hpp>
#অন্তর্ভাব <openvino/op/topk.hpp>
#অন্তর্ভাব <openvino/pass/pattern/op/wrap_type.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ pass {

Fuse_Argsort_শীর্ষ_K::Fuse_Argsort_শীর্ষ_K() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    auto pattern = ov::pass::pattern::wrap_type<ov::op::v11::শীর্ষ_K>();

    ধ্রুব auto callback = [](ov::pass::pattern::Matcher & m) {
        auto topk = ov::as_type_ptr<ov::op::v11::শীর্ষ_K>(m.get_match_মূল());
        যদি (!topk->output(0).get_target_inputs().empty() || topk->output(1).get_target_inputs().empty() ||
            topk->get_sort_type() != ov::op::v11::শীর্ষ_K::SortType::SORT_মানসমূহ) {
            প্রত্যাগম অসত্য;
        }

        ধ্রুব auto shape = topk->get_output_partial_shape(1);
        যদি (shape.rank().is_dynamic()) {
            প্রত্যাগম অসত্য;
        }
        ধ্রুব int64_t rank = shape.rank().get_length();
        ধ্রুব int64_t axis = topk->get_axis();
        যদি (shape[axis].is_dynamic()) {
            প্রত্যাগম অসত্য;
        }

        int64_t prefix = -1;
        প্রতি (ধ্রুব auto & প্রবেশ : topk->output(1).get_target_inputs()) {
            ধ্রুব auto * slice = ov::as_type<ov::op::v8::Slice>(প্রবেশ.get_গ্রন্থি());
            যদি (!slice || প্রবেশ.get_index() != 0 || slice->get_প্রবেশ_size() != 5) {
                প্রত্যাগম অসত্য;
            }
            int64_t মানসমূহ[4];
            প্রতি (size_t i = 0; i < 4; ++i) {
                auto value = ov::as_type_ptr<ov::op::v0::Constant>(slice->get_প্রবেশ_গ্রন্থি_shared_ptr(i + 1));
                যদি (!value || ov::shape_size(value->get_shape()) != 1) {
                    প্রত্যাগম অসত্য;
                }
                মানসমূহ[i] = value->cast_vector<int64_t>()[0];
            }
            ধ্রুব int64_t slice_axis = মানসমূহ[3] < 0 ? মানসমূহ[3] + rank : মানসমূহ[3];
            যদি (মানসমূহ[0] != 0 || মানসমূহ[2] != 1 || slice_axis != axis || মানসমূহ[1] <= 0 ||
                মানসমূহ[1] >= shape[axis].get_length() || (prefix != -1 && prefix != মানসমূহ[1])) {
                প্রত্যাগম অসত্য;
            }
            prefix = মানসমূহ[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->set_argument(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        প্রত্যাগম সত্য;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
