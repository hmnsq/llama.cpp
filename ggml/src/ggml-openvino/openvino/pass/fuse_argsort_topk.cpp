#अन्तर्भाव "fuse_argsort_topk.h"

#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/topk.hpp>
#अन्तर्भाव <openvino/pass/pattern/op/wrap_type.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश pass {

Fuse_Argsort_शिखरम्_K::Fuse_Argsort_शिखरम्_K() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    स्वचालित pattern = ov::pass::pattern::wrap_type<ov::op::v11::शिखरम्_K>();

    ध्रुव स्वचालित callback = [](ov::pass::pattern::Matcher & m) {
        स्वचालित topk = ov::as_type_ptr<ov::op::v11::शिखरम्_K>(m.get_मेलनम्_मूलम्());
        यदि (!topk->output(0).get_target_inputs().रिक्तानि() || topk->output(1).get_target_inputs().रिक्तानि() ||
            topk->get_क्रमय_type() != ov::op::v11::शिखरम्_K::क्रमय_Type::क्रमय_मानानि) {
            प्रत्यागम असत्य;
        }

        ध्रुव स्वचालित shape = topk->get_output_partial_shape(1);
        यदि (shape.rank().is_dynamic()) {
            प्रत्यागम असत्य;
        }
        ध्रुव int64_t rank = shape.rank().get_दीर्घता();
        ध्रुव int64_t axis = topk->get_axis();
        यदि (shape[axis].is_dynamic()) {
            प्रत्यागम असत्य;
        }

        int64_t prefix = -1;
        प्रति (ध्रुव स्वचालित & निवेशः : topk->output(1).get_target_inputs()) {
            ध्रुव स्वचालित * खण्डिका_2 = ov::as_type<ov::op::v8::खण्डिका>(निवेशः.get_ग्रन्थिः());
            यदि (!खण्डिका_2 || निवेशः.get_सूचकांक() != 0 || खण्डिका_2->get_निवेशः_आकार() != 5) {
                प्रत्यागम असत्य;
            }
            int64_t मानानि[4];
            प्रति (size_t i = 0; i < 4; ++i) {
                स्वचालित value = ov::as_type_ptr<ov::op::v0::Constant>(खण्डिका_2->get_निवेशः_ग्रन्थिः_shared_ptr(i + 1));
                यदि (!value || ov::shape_आकार(value->get_shape()) != 1) {
                    प्रत्यागम असत्य;
                }
                मानानि[i] = value->cast_सदिशम्<int64_t>()[0];
            }
            ध्रुव int64_t खण्डिका_axis = मानानि[3] < 0 ? मानानि[3] + rank : मानानि[3];
            यदि (मानानि[0] != 0 || मानानि[2] != 1 || खण्डिका_axis != axis || मानानि[1] <= 0 ||
                मानानि[1] >= shape[axis].get_दीर्घता() || (prefix != -1 && prefix != मानानि[1])) {
                प्रत्यागम असत्य;
            }
            prefix = मानानि[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->समुच्चयः_तर्कः(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        प्रत्यागम सत्य;
    };

    पञ्जिकाकोशः_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
