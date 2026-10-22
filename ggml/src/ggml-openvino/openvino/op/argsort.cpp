#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/frontend/exception.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/squeeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/topk.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_argsort(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);

    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = process_view_ਨਿਵੇਸ਼_new(context, 0);

    ਅਚਲ int32_t ਕ੍ਰਮ = context.get_output_op_params()[0];

    ov::op::v11::ਸਿਖਰ_K::ਬਹੁਲਕ_2 mode;
    ਚੋਣ (ਕ੍ਰਮ) {
    ਹਾਲਤ GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC:
        mode = ov::op::v11::ਸਿਖਰ_K::ਬਹੁਲਕ_2::MIN;
        ਰੋਕੋ;
    ਹਾਲਤ GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC:
        mode = ov::op::v11::ਸਿਖਰ_K::ਬਹੁਲਕ_2::MAX;
        ਰੋਕੋ;
    ਮੂਲ:
        FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(ਝੂਠ, "Unsupported GGML_OP_ARGSORT order: ", ਕ੍ਰਮ);
    }

    // Stateful models drop the leading size-1 batch dim, so the expert axis is 2 there
    // instead of 3 (same rank-3-vs-rank-4 split as get_rows.cpp / process_view_input).
    ਅਚਲ ਪੂਰਨ axis = (context.is_stateful() && ਨਿਵੇਸ਼.get_partial_shape().rank() == 3) ? 2 : 3;

    ਸਵੈਚਾਲਿਤ k = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(ਨਿਵੇਸ਼.get_ਗੰਢ_shared_ptr(), {axis}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    ਸਵੈਚਾਲਿਤ topk = std::make_shared<ov::op::v11::ਸਿਖਰ_K>(ਨਿਵੇਸ਼, k, axis, mode, ov::op::v11::ਸਿਖਰ_K::ਛਾਂਟੋ_Type::ਛਾਂਟੋ_ਮੁੱਲਾਂ,
                                                    context.get_output_type(), ਝੂਠ);

    ਵਾਪਸ rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
