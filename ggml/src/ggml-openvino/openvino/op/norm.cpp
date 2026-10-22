#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/mvn.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_norm(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);

    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_ਗੰਢ = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, context.get_output_op_params(), ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਸਵੈਚਾਲਿਤ axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v6::MVN>(ਨਿਵੇਸ਼_ਗੰਢ, axes, ਸੱਚ, eps, ov::op::MVNEps_ਬਹੁਲਕ::INSIDE_SQRT);

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
