#அந்தர்பாவ "fuse_argsort_topk.h"

#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/slice.hpp>
#அந்தர்பாவ <openvino/op/topk.hpp>
#அந்தர்பாவ <openvino/pass/pattern/op/wrap_type.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ pass {

Fuse_Argsort_உச்சி_K::Fuse_Argsort_உச்சி_K() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    auto pattern = ov::pass::pattern::wrap_type<ov::op::v11::உச்சி_K>();

    மாறிலி auto callback = [](ov::pass::pattern::Matcher & m) {
        auto topk = ov::as_type_ptr<ov::op::v11::உச்சி_K>(m.get_match_வேர்());
        எனில் (!topk->output(0).get_target_inputs().empty() || topk->output(1).get_target_inputs().empty() ||
            topk->get_sort_type() != ov::op::v11::உச்சி_K::SortType::SORT_மதிப்புகள்) {
            திரும்பு பொய்;
        }

        மாறிலி auto shape = topk->get_output_partial_shape(1);
        எனில் (shape.rank().is_dynamic()) {
            திரும்பு பொய்;
        }
        மாறிலி int64_t rank = shape.rank().get_length();
        மாறிலி int64_t axis = topk->get_axis();
        எனில் (shape[axis].is_dynamic()) {
            திரும்பு பொய்;
        }

        int64_t prefix = -1;
        க்கு (மாறிலி auto & உள்ளீடு : topk->output(1).get_target_inputs()) {
            மாறிலி auto * slice = ov::as_type<ov::op::v8::Slice>(உள்ளீடு.get_முனை());
            எனில் (!slice || உள்ளீடு.get_index() != 0 || slice->get_உள்ளீடு_size() != 5) {
                திரும்பு பொய்;
            }
            int64_t மதிப்புகள்[4];
            க்கு (size_t i = 0; i < 4; ++i) {
                auto value = ov::as_type_ptr<ov::op::v0::Constant>(slice->get_உள்ளீடு_முனை_shared_ptr(i + 1));
                எனில் (!value || ov::shape_size(value->get_shape()) != 1) {
                    திரும்பு பொய்;
                }
                மதிப்புகள்[i] = value->cast_vector<int64_t>()[0];
            }
            மாறிலி int64_t slice_axis = மதிப்புகள்[3] < 0 ? மதிப்புகள்[3] + rank : மதிப்புகள்[3];
            எனில் (மதிப்புகள்[0] != 0 || மதிப்புகள்[2] != 1 || slice_axis != axis || மதிப்புகள்[1] <= 0 ||
                மதிப்புகள்[1] >= shape[axis].get_length() || (prefix != -1 && prefix != மதிப்புகள்[1])) {
                திரும்பு பொய்;
            }
            prefix = மதிப்புகள்[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->set_argument(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        திரும்பு உண்மை;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
