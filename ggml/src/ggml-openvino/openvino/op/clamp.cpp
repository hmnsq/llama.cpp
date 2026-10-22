#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <cstring>
#ضمن <openvino/op/clamp.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_ويڪٽر translate_clamp(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);

    خودڪار داخلا_2 = process_view_داخلا_new(context, 0);

    ثابت int32_t * op_params = context.get_output_op_params();
    FRONT_پڇاڙي_CHECK_IMPLEMENTED(op_params != خالي_اشارو, "CLAMP requires output op params");

    عائم min;
    عائم max;
    std::يادگيري_نقل(&min, reinterpret_cast<ثابت عائم *>(op_params) + 0, حجم(عائم));
    std::يادگيري_نقل(&max, reinterpret_cast<ثابت عائم *>(op_params) + 1, حجم(عائم));

    خودڪار res = std::make_shared<ov::op::v0::Clamp>(داخلا_2, min, max);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
