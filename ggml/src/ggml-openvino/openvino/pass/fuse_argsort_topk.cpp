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
    ಸ್ವಯಂಚಾಲಿತ pattern = ov::pass::pattern::wrap_type<ov::op::v11::ತುದಿ_K>();

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ callback = [](ov::pass::pattern::Matcher & m) {
        ಸ್ವಯಂಚಾಲಿತ topk = ov::as_type_ptr<ov::op::v11::ತುದಿ_K>(m.get_ಹೊಂದಿಕೆ_ಬೇರು());
        ಆದರೆ (!topk->output(0).get_ಗುರಿ_inputs().ಖಾಲಿಗಳು() || topk->output(1).get_ಗುರಿ_inputs().ಖಾಲಿಗಳು() ||
            topk->get_ವಿಂಗಡಿಸು_type() != ov::op::v11::ತುದಿ_K::ವಿಂಗಡಿಸು_Type::ವಿಂಗಡಿಸು_ಮೌಲ್ಯಗಳು) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ shape = topk->get_output_partial_shape(1);
        ಆದರೆ (shape.rank().is_dynamic()) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಸ್ಥಿರಾಂಕ int64_t rank = shape.rank().get_ಉದ್ದಳತೆ();
        ಸ್ಥಿರಾಂಕ int64_t axis = topk->get_axis();
        ಆದರೆ (shape[axis].is_dynamic()) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        int64_t prefix = -1;
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಒಳಹರಿವು : topk->output(1).get_ಗುರಿ_inputs()) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ತುಂಡು_2 = ov::as_type<ov::op::v8::ತುಂಡು>(ಒಳಹರಿವು.get_ಗಂಟು());
            ಆದರೆ (!ತುಂಡು_2 || ಒಳಹರಿವು.get_ಸೂಚ್ಯಂಕ() != 0 || ತುಂಡು_2->get_ಒಳಹರಿವು_ಗಾತ್ರ() != 5) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            int64_t ಮೌಲ್ಯಗಳು[4];
            ಗಾಗಿ (size_t i = 0; i < 4; ++i) {
                ಸ್ವಯಂಚಾಲಿತ value = ov::as_type_ptr<ov::op::v0::Constant>(ತುಂಡು_2->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(i + 1));
                ಆದರೆ (!value || ov::shape_ಗಾತ್ರ(value->get_shape()) != 1) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಮೌಲ್ಯಗಳು[i] = value->cast_ಸದಿಶ<int64_t>()[0];
            }
            ಸ್ಥಿರಾಂಕ int64_t ತುಂಡು_axis = ಮೌಲ್ಯಗಳು[3] < 0 ? ಮೌಲ್ಯಗಳು[3] + rank : ಮೌಲ್ಯಗಳು[3];
            ಆದರೆ (ಮೌಲ್ಯಗಳು[0] != 0 || ಮೌಲ್ಯಗಳು[2] != 1 || ತುಂಡು_axis != axis || ಮೌಲ್ಯಗಳು[1] <= 0 ||
                ಮೌಲ್ಯಗಳು[1] >= shape[axis].get_ಉದ್ದಳತೆ() || (prefix != -1 && prefix != ಮೌಲ್ಯಗಳು[1])) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            prefix = ಮೌಲ್ಯಗಳು[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->ಗಣ_ವಾದ(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        ಹಿಂತಿರುಗಿ ನಿಜ;
    };

    ರಿಜಿಸ್ಟರ್_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
