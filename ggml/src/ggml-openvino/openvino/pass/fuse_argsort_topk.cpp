#ਅਨ੍ਤਰ੍ਭਾਵ "fuse_argsort_topk.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/slice.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/topk.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/pass/pattern/op/wrap_type.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ pass {

Fuse_Argsort_ਸਿਖਰ_K::Fuse_Argsort_ਸਿਖਰ_K() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    ਸਵੈਚਾਲਿਤ pattern = ov::pass::pattern::wrap_type<ov::op::v11::ਸਿਖਰ_K>();

    ਅਚਲ ਸਵੈਚਾਲਿਤ callback = [](ov::pass::pattern::Matcher & m) {
        ਸਵੈਚਾਲਿਤ topk = ov::as_type_ptr<ov::op::v11::ਸਿਖਰ_K>(m.get_ਮੇਲ_ਜੜ੍ਹ());
        ਜੇ (!topk->output(0).get_ਨਿਸ਼ਾਨਾ_inputs().ਖਾਲੀ_ਥਾਵਾਂ() || topk->output(1).get_ਨਿਸ਼ਾਨਾ_inputs().ਖਾਲੀ_ਥਾਵਾਂ() ||
            topk->get_ਛਾਂਟੋ_type() != ov::op::v11::ਸਿਖਰ_K::ਛਾਂਟੋ_Type::ਛਾਂਟੋ_ਮੁੱਲਾਂ) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ਸਵੈਚਾਲਿਤ shape = topk->get_output_partial_shape(1);
        ਜੇ (shape.rank().is_dynamic()) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਅਚਲ int64_t rank = shape.rank().get_ਲੰਮਾਈ();
        ਅਚਲ int64_t axis = topk->get_axis();
        ਜੇ (shape[axis].is_dynamic()) {
            ਵਾਪਸ ਝੂਠ;
        }

        int64_t prefix = -1;
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ਨਿਵੇਸ਼ : topk->output(1).get_ਨਿਸ਼ਾਨਾ_inputs()) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ * ਟੁਕੜਾ_2 = ov::as_type<ov::op::v8::ਟੁਕੜਾ>(ਨਿਵੇਸ਼.get_ਗੰਢ());
            ਜੇ (!ਟੁਕੜਾ_2 || ਨਿਵੇਸ਼.get_ਸੂਚਕਾਂਕ() != 0 || ਟੁਕੜਾ_2->get_ਨਿਵੇਸ਼_ਮਾਪ() != 5) {
                ਵਾਪਸ ਝੂਠ;
            }
            int64_t ਮੁੱਲਾਂ[4];
            ਲਈ (size_t i = 0; i < 4; ++i) {
                ਸਵੈਚਾਲਿਤ value = ov::as_type_ptr<ov::op::v0::Constant>(ਟੁਕੜਾ_2->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(i + 1));
                ਜੇ (!value || ov::shape_ਮਾਪ(value->get_shape()) != 1) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਮੁੱਲਾਂ[i] = value->cast_ਸਦਿਸ਼<int64_t>()[0];
            }
            ਅਚਲ int64_t ਟੁਕੜਾ_axis = ਮੁੱਲਾਂ[3] < 0 ? ਮੁੱਲਾਂ[3] + rank : ਮੁੱਲਾਂ[3];
            ਜੇ (ਮੁੱਲਾਂ[0] != 0 || ਮੁੱਲਾਂ[2] != 1 || ਟੁਕੜਾ_axis != axis || ਮੁੱਲਾਂ[1] <= 0 ||
                ਮੁੱਲਾਂ[1] >= shape[axis].get_ਲੰਮਾਈ() || (prefix != -1 && prefix != ਮੁੱਲਾਂ[1])) {
                ਵਾਪਸ ਝੂਠ;
            }
            prefix = ਮੁੱਲਾਂ[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->ਸੈੱਟ_ਦਲੀਲ(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        ਵਾਪਸ ਸੱਚ;
    };

    ਰਜਿਸਟਰ_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
