#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <cstring>
#ضمن <openvino/op/clamp.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_سمتیہ translate_clamp(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    خودکار داخلہ = process_view_داخلہ_new(context, 0);

    ثابت int32_t * op_params = context.get_output_op_params();
    FRONT_سرا_CHECK_IMPLEMENTED(op_params != خالی_سوچک, "CLAMP requires output op params");

    عائم min;
    عائم max;
    std::حافظہ_نقل(&min, reinterpret_cast<ثابت عائم *>(op_params) + 0, حجم(عائم));
    std::حافظہ_نقل(&max, reinterpret_cast<ثابت عائم *>(op_params) + 1, حجم(عائم));

    خودکار res = std::make_shared<ov::op::v0::Clamp>(داخلہ, min, max);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
