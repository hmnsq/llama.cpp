#അന്തര്ഭാവ "fuse_argsort_topk.h"

#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/slice.hpp>
#അന്തര്ഭാവ <openvino/op/topk.hpp>
#അന്തര്ഭാവ <openvino/pass/pattern/op/wrap_type.hpp>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ pass {

Fuse_Argsort_മുകൾ_K::Fuse_Argsort_മുകൾ_K() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    auto pattern = ov::pass::pattern::wrap_type<ov::op::v11::മുകൾ_K>();

    സ്ഥിരം auto callback = [](ov::pass::pattern::Matcher & m) {
        auto topk = ov::as_type_ptr<ov::op::v11::മുകൾ_K>(m.get_match_വേര്());
        എങ്കിൽ (!topk->output(0).get_target_inputs().empty() || topk->output(1).get_target_inputs().empty() ||
            topk->get_sort_type() != ov::op::v11::മുകൾ_K::SortType::SORT_മൂല്യങ്ങൾ) {
            തിരികെ അസത്യം;
        }

        സ്ഥിരം auto shape = topk->get_output_partial_shape(1);
        എങ്കിൽ (shape.rank().is_dynamic()) {
            തിരികെ അസത്യം;
        }
        സ്ഥിരം int64_t rank = shape.rank().get_length();
        സ്ഥിരം int64_t axis = topk->get_axis();
        എങ്കിൽ (shape[axis].is_dynamic()) {
            തിരികെ അസത്യം;
        }

        int64_t prefix = -1;
        വേണ്ടി (സ്ഥിരം auto & നിവേശം : topk->output(1).get_target_inputs()) {
            സ്ഥിരം auto * slice = ov::as_type<ov::op::v8::Slice>(നിവേശം.get_കണ്ണി());
            എങ്കിൽ (!slice || നിവേശം.get_index() != 0 || slice->get_നിവേശം_size() != 5) {
                തിരികെ അസത്യം;
            }
            int64_t മൂല്യങ്ങൾ[4];
            വേണ്ടി (size_t i = 0; i < 4; ++i) {
                auto value = ov::as_type_ptr<ov::op::v0::Constant>(slice->get_നിവേശം_കണ്ണി_shared_ptr(i + 1));
                എങ്കിൽ (!value || ov::shape_size(value->get_shape()) != 1) {
                    തിരികെ അസത്യം;
                }
                മൂല്യങ്ങൾ[i] = value->cast_vector<int64_t>()[0];
            }
            സ്ഥിരം int64_t slice_axis = മൂല്യങ്ങൾ[3] < 0 ? മൂല്യങ്ങൾ[3] + rank : മൂല്യങ്ങൾ[3];
            എങ്കിൽ (മൂല്യങ്ങൾ[0] != 0 || മൂല്യങ്ങൾ[2] != 1 || slice_axis != axis || മൂല്യങ്ങൾ[1] <= 0 ||
                മൂല്യങ്ങൾ[1] >= shape[axis].get_length() || (prefix != -1 && prefix != മൂല്യങ്ങൾ[1])) {
                തിരികെ അസത്യം;
            }
            prefix = മൂല്യങ്ങൾ[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->set_argument(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        തിരികെ സത്യം;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
