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

Output_وکتور translate_concat(ثابت گره_Context & context) {
    رقم_inputs_check(context, 2, 2);

    ثابت int32_t * op_params = context.get_output_op_params();
    FRONT_انتها_CHECK_IMPLEMENTED(op_params != اشاره_تهی, "CONCAT requires output op params");

    ثابت خودکار output_shape = context.get_output_shape();
    FRONT_انتها_CHECK_IMPLEMENTED(output_shape.rank().is_static(), "CONCAT requires static output rank");

    ثابت خودکار rank = output_shape.rank().get_درازی();
    ثابت int32_t ggml_dim = op_params[0];
    FRONT_انتها_CHECK_IMPLEMENTED(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    خودکار درون_داد_0 = process_view_درون_داد_new(context, 0);
    خودکار درون_داد_1 = process_view_درون_داد_new(context, 1);
    ثابت خودکار output_type = context.get_output_type();

    إذا (درون_داد_0.get_element_type() != output_type) {
        درون_داد_0 = std::make_shared<ov::op::v0::Convert>(درون_داد_0, output_type);
    }
    إذا (درون_داد_1.get_element_type() != output_type) {
        درون_داد_1 = std::make_shared<ov::op::v0::Convert>(درون_داد_1, output_type);
    }

    ثابت خودکار axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    خودکار res = std::make_shared<ov::op::v0::Concat>(Output_وکتور{درون_داد_0, درون_داد_1}, axis);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
