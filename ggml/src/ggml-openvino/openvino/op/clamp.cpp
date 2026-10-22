#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <cstring>
#شامل <openvino/op/clamp.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

OutputVector translate_clamp(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    auto درون_داد = process_view_درون_داد_new(context, 0);

    ثابت int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CLAMP requires output op params");

    اعشاری min;
    اعشاری max;
    std::memcpy(&min, reinterpret_cast<ثابت اعشاری *>(op_params) + 0, اندازه(اعشاری));
    std::memcpy(&max, reinterpret_cast<ثابت اعشاری *>(op_params) + 1, اندازه(اعشاری));

    auto res = std::make_shared<ov::op::v0::Clamp>(درون_داد, min, max);
    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
