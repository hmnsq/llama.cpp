#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <memory>
#شامل <openvino/frontend/exception.hpp>
#شامل <openvino/op/concat.hpp>
#شامل <openvino/op/convert.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_concat(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 2, 2);

    ثابت int32_t * op_params = context.get_output_op_params();
    FRONT_انتہا_CHECK_IMPLEMENTED(op_params != خالی_اشاریہ, "CONCAT requires output op params");

    ثابت خودکار output_shape = context.get_output_shape();
    FRONT_انتہا_CHECK_IMPLEMENTED(output_shape.rank().is_static(), "CONCAT requires static output rank");

    ثابت خودکار rank = output_shape.rank().get_لمبائی();
    ثابت int32_t ggml_dim = op_params[0];
    FRONT_انتہا_CHECK_IMPLEMENTED(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    خودکار داخلہ_0 = process_view_داخلہ_new(context, 0);
    خودکار داخلہ_1 = process_view_داخلہ_new(context, 1);
    ثابت خودکار output_type = context.get_output_type();

    اگر (داخلہ_0.get_element_type() != output_type) {
        داخلہ_0 = std::make_shared<ov::op::v0::Convert>(داخلہ_0, output_type);
    }
    اگر (داخلہ_1.get_element_type() != output_type) {
        داخلہ_1 = std::make_shared<ov::op::v0::Convert>(داخلہ_1, output_type);
    }

    ثابت خودکار axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    خودکار res = std::make_shared<ov::op::v0::Concat>(Output_سمتیہ{داخلہ_0, داخلہ_1}, axis);

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
