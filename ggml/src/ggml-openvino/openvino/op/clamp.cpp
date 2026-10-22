#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/clamp.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_clamp(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);

    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = process_view_ਨਿਵੇਸ਼_new(context, 0);

    ਅਚਲ int32_t * op_params = context.get_output_op_params();
    FRONT_ਸਿਰਾ_CHECK_IMPLEMENTED(op_params != ਖਾਲੀ_ਸੂਚਕ, "CLAMP requires output op params");

    ਦਸ਼ਮਲਵ min;
    ਦਸ਼ਮਲਵ max;
    std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(&min, reinterpret_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(op_params) + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max, reinterpret_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(op_params) + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v0::Clamp>(ਨਿਵੇਸ਼, min, max);
    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
