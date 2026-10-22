#अन्तर्भाव "fuse_argsort_topk.h"

#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/topk.hpp>
#अन्तर्भाव <openvino/pass/pattern/op/wrap_type.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश pass {

Fuse_Argsort_शीर्ष_K::Fuse_Argsort_शीर्ष_K() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    स्वतः pattern = ov::pass::pattern::wrap_type<ov::op::v11::शीर्ष_K>();

    स्थिरांक स्वतः callback = [](ov::pass::pattern::Matcher & m) {
        स्वतः topk = ov::as_type_ptr<ov::op::v11::शीर्ष_K>(m.get_match_मूळ());
        जर (!topk->output(0).get_target_inputs().empty() || topk->output(1).get_target_inputs().empty() ||
            topk->get_sort_type() != ov::op::v11::शीर्ष_K::SortType::SORT_मूल्ये) {
            परत असत्य;
        }

        स्थिरांक स्वतः shape = topk->get_output_partial_shape(1);
        जर (shape.rank().is_dynamic()) {
            परत असत्य;
        }
        स्थिरांक int64_t rank = shape.rank().get_length();
        स्थिरांक int64_t axis = topk->get_axis();
        जर (shape[axis].is_dynamic()) {
            परत असत्य;
        }

        int64_t prefix = -1;
        साठी (स्थिरांक स्वतः & आदान : topk->output(1).get_target_inputs()) {
            स्थिरांक स्वतः * slice = ov::as_type<ov::op::v8::Slice>(आदान.get_गाठ());
            जर (!slice || आदान.get_index() != 0 || slice->get_आदान_size() != 5) {
                परत असत्य;
            }
            int64_t मूल्ये[4];
            साठी (माप_प्रकार i = 0; i < 4; ++i) {
                स्वतः value = ov::as_type_ptr<ov::op::v0::Constant>(slice->get_आदान_गाठ_shared_ptr(i + 1));
                जर (!value || ov::shape_size(value->get_shape()) != 1) {
                    परत असत्य;
                }
                मूल्ये[i] = value->cast_vector<int64_t>()[0];
            }
            स्थिरांक int64_t slice_axis = मूल्ये[3] < 0 ? मूल्ये[3] + rank : मूल्ये[3];
            जर (मूल्ये[0] != 0 || मूल्ये[2] != 1 || slice_axis != axis || मूल्ये[1] <= 0 ||
                मूल्ये[1] >= shape[axis].get_length() || (prefix != -1 && prefix != मूल्ये[1])) {
                परत असत्य;
            }
            prefix = मूल्ये[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->set_argument(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        परत सत्य;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
