#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <cstring>
#ضمن <openvino/op/clamp.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_وکتور translate_clamp(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    خودکار درون_داد = process_view_درون_داد_new(context, 0);

    ثابت int32_t * op_params = context.get_output_op_params();
    FRONT_انتها_CHECK_IMPLEMENTED(op_params != اشاره_تهی, "CLAMP requires output op params");

    عائم min;
    عائم max;
    std::کپی_حافظه(&min, reinterpret_cast<ثابت عائم *>(op_params) + 0, حجم(عائم));
    std::کپی_حافظه(&max, reinterpret_cast<ثابت عائم *>(op_params) + 1, حجم(عائم));

    خودکار res = std::make_shared<ov::op::v0::Clamp>(درون_داد, min, max);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
