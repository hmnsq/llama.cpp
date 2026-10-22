#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <memory>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/sqrt.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_sqr(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    خودکار داخلہ = process_view_داخلہ_new(context, 0);
    خودکار res = std::make_shared<ov::op::v1::Multiply>(داخلہ, داخلہ);

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

Output_سمتیہ translate_sqrt(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    خودکار داخلہ = process_view_داخلہ_new(context, 0);
    خودکار res = std::make_shared<ov::op::v0::Sqrt>(داخلہ);

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
