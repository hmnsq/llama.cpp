#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <cstring>
#ضمن <openvino/op/clamp.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

OutputVector translate_clamp(ثابت NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input = process_view_input_new(context, 0);

    ثابت int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CLAMP requires output op params");

    عائم min;
    عائم max;
    std::memcpy(&min, reinterpret_cast<ثابت عائم *>(op_params) + 0, حجم(عائم));
    std::memcpy(&max, reinterpret_cast<ثابت عائم *>(op_params) + 1, حجم(عائم));

    auto res = std::make_shared<ov::op::v0::Clamp>(input, min, max);
    ܦܢܐ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
