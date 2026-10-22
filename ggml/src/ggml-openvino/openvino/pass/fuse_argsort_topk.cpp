#ଅନ୍ତର୍ଭାଵ "fuse_argsort_topk.h"

#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/slice.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/topk.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/pass/pattern/op/wrap_type.hpp>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ pass {

Fuse_Argsort_ଶୀର୍ଷ_K::Fuse_Argsort_ଶୀର୍ଷ_K() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    auto pattern = ov::pass::pattern::wrap_type<ov::op::v11::ଶୀର୍ଷ_K>();

    ଅଚଳ auto callback = [](ov::pass::pattern::Matcher & m) {
        auto topk = ov::as_type_ptr<ov::op::v11::ଶୀର୍ଷ_K>(m.get_match_ମୂଳ());
        ଯଦି (!topk->output(0).get_target_inputs().empty() || topk->output(1).get_target_inputs().empty() ||
            topk->get_sort_type() != ov::op::v11::ଶୀର୍ଷ_K::SortType::SORT_ମୂଲ୍ୟଗୁଡ଼ିକ) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ auto shape = topk->get_output_partial_shape(1);
        ଯଦି (shape.rank().is_dynamic()) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଅଚଳ int64_t rank = shape.rank().get_length();
        ଅଚଳ int64_t axis = topk->get_axis();
        ଯଦି (shape[axis].is_dynamic()) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        int64_t prefix = -1;
        ପାଇଁ (ଅଚଳ auto & ନିବେଶ : topk->output(1).get_target_inputs()) {
            ଅଚଳ auto * slice = ov::as_type<ov::op::v8::Slice>(ନିବେଶ.get_ଗଣ୍ଠି());
            ଯଦି (!slice || ନିବେଶ.get_index() != 0 || slice->get_ନିବେଶ_size() != 5) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            int64_t ମୂଲ୍ୟଗୁଡ଼ିକ[4];
            ପାଇଁ (size_t i = 0; i < 4; ++i) {
                auto value = ov::as_type_ptr<ov::op::v0::Constant>(slice->get_ନିବେଶ_ଗଣ୍ଠି_shared_ptr(i + 1));
                ଯଦି (!value || ov::shape_size(value->get_shape()) != 1) {
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }
                ମୂଲ୍ୟଗୁଡ଼ିକ[i] = value->cast_vector<int64_t>()[0];
            }
            ଅଚଳ int64_t slice_axis = ମୂଲ୍ୟଗୁଡ଼ିକ[3] < 0 ? ମୂଲ୍ୟଗୁଡ଼ିକ[3] + rank : ମୂଲ୍ୟଗୁଡ଼ିକ[3];
            ଯଦି (ମୂଲ୍ୟଗୁଡ଼ିକ[0] != 0 || ମୂଲ୍ୟଗୁଡ଼ିକ[2] != 1 || slice_axis != axis || ମୂଲ୍ୟଗୁଡ଼ିକ[1] <= 0 ||
                ମୂଲ୍ୟଗୁଡ଼ିକ[1] >= shape[axis].get_length() || (prefix != -1 && prefix != ମୂଲ୍ୟଗୁଡ଼ିକ[1])) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            prefix = ମୂଲ୍ୟଗୁଡ଼ିକ[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->set_argument(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        ଫେରନ୍ତୁ ସତ୍ୟ;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
