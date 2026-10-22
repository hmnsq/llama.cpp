#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <cstring>
#અન્તર્ભાવ <openvino/op/clamp.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_clamp(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);

    auto આગત = process_view_આગત_new(context, 0);

    અચલ int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CLAMP requires output op params");

    દશાંશ min;
    દશાંશ max;
    std::memcpy(&min, reinterpret_cast<અચલ દશાંશ *>(op_params) + 0, કદ(દશાંશ));
    std::memcpy(&max, reinterpret_cast<અચલ દશાંશ *>(op_params) + 1, કદ(દશાંશ));

    auto res = std::make_shared<ov::op::v0::Clamp>(આગત, min, max);
    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
