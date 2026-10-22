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

    अचर स्वतः callback = [](ov::pass::pattern::Matcher & m) {
        स्वतः topk = ov::as_type_ptr<ov::op::v11::शीर्ष_K>(m.get_match_जरा());
        यदि (!topk->output(0).get_target_inputs().empty() || topk->output(1).get_target_inputs().empty() ||
            topk->get_sort_type() != ov::op::v11::शीर्ष_K::SortType::SORT_मानहरू) {
            फर्काऊ असत्य;
        }

        अचर स्वतः shape = topk->get_output_partial_shape(1);
        यदि (shape.rank().is_dynamic()) {
            फर्काऊ असत्य;
        }
        अचर int64_t rank = shape.rank().get_length();
        अचर int64_t axis = topk->get_axis();
        यदि (shape[axis].is_dynamic()) {
            फर्काऊ असत्य;
        }

        int64_t prefix = -1;
        लागि (अचर स्वतः & आगत : topk->output(1).get_target_inputs()) {
            अचर स्वतः * slice = ov::as_type<ov::op::v8::Slice>(आगत.get_गाँठो());
            यदि (!slice || आगत.get_index() != 0 || slice->get_आगत_size() != 5) {
                फर्काऊ असत्य;
            }
            int64_t मानहरू[4];
            लागि (माप_प्रकार i = 0; i < 4; ++i) {
                स्वतः value = ov::as_type_ptr<ov::op::v0::Constant>(slice->get_आगत_गाँठो_shared_ptr(i + 1));
                यदि (!value || ov::shape_size(value->get_shape()) != 1) {
                    फर्काऊ असत्य;
                }
                मानहरू[i] = value->cast_vector<int64_t>()[0];
            }
            अचर int64_t slice_axis = मानहरू[3] < 0 ? मानहरू[3] + rank : मानहरू[3];
            यदि (मानहरू[0] != 0 || मानहरू[2] != 1 || slice_axis != axis || मानहरू[1] <= 0 ||
                मानहरू[1] >= shape[axis].get_length() || (prefix != -1 && prefix != मानहरू[1])) {
                फर्काऊ असत्य;
            }
            prefix = मानहरू[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->set_argument(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        फर्काऊ सत्य;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
