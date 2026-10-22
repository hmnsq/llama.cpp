#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <cstring>
#شامل <openvino/op/clamp.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_clamp(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    خودکار داخلہ = process_view_داخلہ_new(context, 0);

    ثابت int32_t * op_params = context.get_output_op_params();
    FRONT_انتہا_CHECK_IMPLEMENTED(op_params != خالی_اشاریہ, "CLAMP requires output op params");

    اعشاری min;
    اعشاری max;
    std::حافظہ_نقل(&min, reinterpret_cast<ثابت اعشاری *>(op_params) + 0, حجم(اعشاری));
    std::حافظہ_نقل(&max, reinterpret_cast<ثابت اعشاری *>(op_params) + 1, حجم(اعشاری));

    خودکار res = std::make_shared<ov::op::v0::Clamp>(داخلہ, min, max);
    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
