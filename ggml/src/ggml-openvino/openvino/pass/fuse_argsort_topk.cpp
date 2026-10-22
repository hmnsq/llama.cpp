#समावेश "fuse_argsort_topk.h"

#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/slice.hpp>
#समावेश <openvino/op/topk.hpp>
#समावेश <openvino/pass/pattern/op/wrap_type.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

Fuse_Argsort_शीर्ष_K::Fuse_Argsort_शीर्ष_K() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    स्वतः pattern = ov::pass::pattern::wrap_type<ov::op::v11::शीर्ष_K>();

    स्थिर स्वतः callback = [](ov::pass::pattern::Matcher & m) {
        स्वतः topk = ov::as_type_ptr<ov::op::v11::शीर्ष_K>(m.get_match_मूल());
        अगर (!topk->output(0).get_target_inputs().empty() || topk->output(1).get_target_inputs().empty() ||
            topk->get_sort_type() != ov::op::v11::शीर्ष_K::SortType::SORT_मूल्यावली) {
            वापस false;
        }

        स्थिर स्वतः shape = topk->get_output_partial_shape(1);
        अगर (shape.rank().is_dynamic()) {
            वापस false;
        }
        स्थिर int64_t rank = shape.rank().get_लंबाई();
        स्थिर int64_t axis = topk->get_axis();
        अगर (shape[axis].is_dynamic()) {
            वापस false;
        }

        int64_t prefix = -1;
        क्रम (स्थिर स्वतः & निवेश : topk->output(1).get_target_inputs()) {
            स्थिर स्वतः * slice = ov::as_type<ov::op::v8::Slice>(निवेश.get_गाँठ());
            अगर (!slice || निवेश.get_सूचकांक() != 0 || slice->get_निवेश_आकार() != 5) {
                वापस false;
            }
            int64_t मूल्यावली[4];
            क्रम (माप_प्रकार i = 0; i < 4; ++i) {
                स्वतः value = ov::as_type_ptr<ov::op::v0::Constant>(slice->get_निवेश_गाँठ_shared_ptr(i + 1));
                अगर (!value || ov::shape_आकार(value->get_shape()) != 1) {
                    वापस false;
                }
                मूल्यावली[i] = value->cast_vector<int64_t>()[0];
            }
            स्थिर int64_t slice_axis = मूल्यावली[3] < 0 ? मूल्यावली[3] + rank : मूल्यावली[3];
            अगर (मूल्यावली[0] != 0 || मूल्यावली[2] != 1 || slice_axis != axis || मूल्यावली[1] <= 0 ||
                मूल्यावली[1] >= shape[axis].get_लंबाई() || (prefix != -1 && prefix != मूल्यावली[1])) {
                वापस false;
            }
            prefix = मूल्यावली[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->set_argument(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        वापस true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
