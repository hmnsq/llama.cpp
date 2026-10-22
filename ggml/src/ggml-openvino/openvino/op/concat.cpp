#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <memory>
#ضمن <openvino/frontend/exception.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/convert.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

OutputVector translate_concat(ثابت كلي_Context & context) {
    لمب_inputs_check(context, 2, 2);

    ثابت int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CONCAT requires output op params");

    ثابت auto output_shape = context.get_output_shape();
    FRONT_END_CHECK_IMPLEMENTED(output_shape.rank().is_static(), "CONCAT requires static output rank");

    ثابت auto rank = output_shape.rank().get_length();
    ثابت int32_t ggml_dim = op_params[0];
    FRONT_END_CHECK_IMPLEMENTED(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    auto شغرو_0 = process_view_شغرو_new(context, 0);
    auto شغرو_1 = process_view_شغرو_new(context, 1);
    ثابت auto output_type = context.get_output_type();

    إذا (شغرو_0.get_element_type() != output_type) {
        شغرو_0 = std::make_shared<ov::op::v0::Convert>(شغرو_0, output_type);
    }
    إذا (شغرو_1.get_element_type() != output_type) {
        شغرو_1 = std::make_shared<ov::op::v0::Convert>(شغرو_1, output_type);
    }

    ثابت auto axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    auto res = std::make_shared<ov::op::v0::Concat>(OutputVector{شغرو_0, شغرو_1}, axis);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
