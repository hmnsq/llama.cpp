#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <memory>
#شامل <openvino/frontend/exception.hpp>
#شامل <openvino/op/concat.hpp>
#شامل <openvino/op/convert.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

OutputVector translate_concat(ثابت گره_Context & context) {
    رقم_inputs_check(context, 2, 2);

    ثابت int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CONCAT requires output op params");

    ثابت auto output_shape = context.get_output_shape();
    FRONT_END_CHECK_IMPLEMENTED(output_shape.rank().is_static(), "CONCAT requires static output rank");

    ثابت auto rank = output_shape.rank().get_length();
    ثابت int32_t ggml_dim = op_params[0];
    FRONT_END_CHECK_IMPLEMENTED(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    auto درون_داد_0 = process_view_درون_داد_new(context, 0);
    auto درون_داد_1 = process_view_درون_داد_new(context, 1);
    ثابت auto output_type = context.get_output_type();

    اگر (درون_داد_0.get_element_type() != output_type) {
        درون_داد_0 = std::make_shared<ov::op::v0::Convert>(درون_داد_0, output_type);
    }
    اگر (درون_داد_1.get_element_type() != output_type) {
        درون_داد_1 = std::make_shared<ov::op::v0::Convert>(درون_داد_1, output_type);
    }

    ثابت auto axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    auto res = std::make_shared<ov::op::v0::Concat>(OutputVector{درون_داد_0, درون_داد_1}, axis);

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
