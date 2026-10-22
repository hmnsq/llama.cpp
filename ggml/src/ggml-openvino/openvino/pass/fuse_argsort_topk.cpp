#અન્તર્ભાવ "fuse_argsort_topk.h"

#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/slice.hpp>
#અન્તર્ભાવ <openvino/op/topk.hpp>
#અન્તર્ભાવ <openvino/pass/pattern/op/wrap_type.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ pass {

Fuse_Argsort_ટોચ_K::Fuse_Argsort_ટોચ_K() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    auto pattern = ov::pass::pattern::wrap_type<ov::op::v11::ટોચ_K>();

    અચલ auto callback = [](ov::pass::pattern::Matcher & m) {
        auto topk = ov::as_type_ptr<ov::op::v11::ટોચ_K>(m.get_match_મૂળ());
        જો (!topk->output(0).get_target_inputs().empty() || topk->output(1).get_target_inputs().empty() ||
            topk->get_sort_type() != ov::op::v11::ટોચ_K::SortType::SORT_કિંમતો) {
            પાછા ખોટું;
        }

        અચલ auto shape = topk->get_output_partial_shape(1);
        જો (shape.rank().is_dynamic()) {
            પાછા ખોટું;
        }
        અચલ int64_t rank = shape.rank().get_length();
        અચલ int64_t axis = topk->get_axis();
        જો (shape[axis].is_dynamic()) {
            પાછા ખોટું;
        }

        int64_t prefix = -1;
        માટે (અચલ auto & આગત : topk->output(1).get_target_inputs()) {
            અચલ auto * slice = ov::as_type<ov::op::v8::Slice>(આગત.get_ગાંઠ());
            જો (!slice || આગત.get_index() != 0 || slice->get_આગત_size() != 5) {
                પાછા ખોટું;
            }
            int64_t કિંમતો[4];
            માટે (size_t i = 0; i < 4; ++i) {
                auto value = ov::as_type_ptr<ov::op::v0::Constant>(slice->get_આગત_ગાંઠ_shared_ptr(i + 1));
                જો (!value || ov::shape_size(value->get_shape()) != 1) {
                    પાછા ખોટું;
                }
                કિંમતો[i] = value->cast_vector<int64_t>()[0];
            }
            અચલ int64_t slice_axis = કિંમતો[3] < 0 ? કિંમતો[3] + rank : કિંમતો[3];
            જો (કિંમતો[0] != 0 || કિંમતો[2] != 1 || slice_axis != axis || કિંમતો[1] <= 0 ||
                કિંમતો[1] >= shape[axis].get_length() || (prefix != -1 && prefix != કિંમતો[1])) {
                પાછા ખોટું;
            }
            prefix = કિંમતો[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->set_argument(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        પાછા સાચું;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
