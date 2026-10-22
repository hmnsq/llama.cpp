#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/broadcast.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

// GGML FILL sets all elements of a tensor to a constant value.
// The constant is stored as a float in op_params[0].
Output_ਸਦਿਸ਼ translate_fill(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);

    ਦਸ਼ਮਲਵ c;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&c, context.get_output_op_params(), ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਸਵੈਚਾਲਿਤ shape = context.get_ਨਿਵੇਸ਼_shape(0).ਮੰਜ਼ਿਲ_shape();

    ਸਵੈਚਾਲਿਤ val = ov::op::v0::Constant::create(context.get_output_type(), {}, {c});
    ਸਵੈਚਾਲਿਤ ਨਿਸ਼ਾਨਾ_shape = ov::op::v0::Constant::create(ov::element::i64, {shape.size()},
        std::vector<int64_t>(shape.begin(), shape.end()));
    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v3::ਪ੍ਰਸਾਰਣ>(val, ਨਿਸ਼ਾਨਾ_shape);

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
