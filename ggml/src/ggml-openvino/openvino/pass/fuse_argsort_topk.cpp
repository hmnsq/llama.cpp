#ಅನ್ತರ್ಭಾವ "fuse_argsort_topk.h"

#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/topk.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/pass/pattern/op/wrap_type.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ pass {

Fuse_Argsort_ತುದಿ_K::Fuse_Argsort_ತುದಿ_K() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    auto pattern = ov::pass::pattern::wrap_type<ov::op::v11::ತುದಿ_K>();

    ಧ್ರುವ auto callback = [](ov::pass::pattern::Matcher & m) {
        auto topk = ov::as_type_ptr<ov::op::v11::ತುದಿ_K>(m.get_match_ಬೇರು());
        ಯದಿ (!topk->output(0).get_target_inputs().empty() || topk->output(1).get_target_inputs().empty() ||
            topk->get_sort_type() != ov::op::v11::ತುದಿ_K::SortType::SORT_ಮೌಲ್ಯಗಳು) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ auto shape = topk->get_output_partial_shape(1);
        ಯದಿ (shape.rank().is_dynamic()) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಧ್ರುವ int64_t rank = shape.rank().get_length();
        ಧ್ರುವ int64_t axis = topk->get_axis();
        ಯದಿ (shape[axis].is_dynamic()) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        int64_t prefix = -1;
        ಪ್ರತಿ (ಧ್ರುವ auto & ಒಳಹರಿವು : topk->output(1).get_target_inputs()) {
            ಧ್ರುವ auto * slice = ov::as_type<ov::op::v8::Slice>(ಒಳಹರಿವು.get_ಗಂಟು());
            ಯದಿ (!slice || ಒಳಹರಿವು.get_index() != 0 || slice->get_ಒಳಹರಿವು_size() != 5) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            int64_t ಮೌಲ್ಯಗಳು[4];
            ಪ್ರತಿ (size_t i = 0; i < 4; ++i) {
                auto value = ov::as_type_ptr<ov::op::v0::Constant>(slice->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(i + 1));
                ಯದಿ (!value || ov::shape_size(value->get_shape()) != 1) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಮೌಲ್ಯಗಳು[i] = value->cast_vector<int64_t>()[0];
            }
            ಧ್ರುವ int64_t slice_axis = ಮೌಲ್ಯಗಳು[3] < 0 ? ಮೌಲ್ಯಗಳು[3] + rank : ಮೌಲ್ಯಗಳು[3];
            ಯದಿ (ಮೌಲ್ಯಗಳು[0] != 0 || ಮೌಲ್ಯಗಳು[2] != 1 || slice_axis != axis || ಮೌಲ್ಯಗಳು[1] <= 0 ||
                ಮೌಲ್ಯಗಳು[1] >= shape[axis].get_length() || (prefix != -1 && prefix != ಮೌಲ್ಯಗಳು[1])) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            prefix = ಮೌಲ್ಯಗಳು[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->set_argument(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
