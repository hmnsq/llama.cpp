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

Output_ويڪٽر translate_concat(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 2, 2);

    ثابت int32_t * op_params = context.get_output_op_params();
    FRONT_پڇاڙي_CHECK_IMPLEMENTED(op_params != خالي_اشارو, "CONCAT requires output op params");

    ثابت خودڪار output_shape = context.get_output_shape();
    FRONT_پڇاڙي_CHECK_IMPLEMENTED(output_shape.rank().is_static(), "CONCAT requires static output rank");

    ثابت خودڪار rank = output_shape.rank().get_ڊيگهائي();
    ثابت int32_t ggml_dim = op_params[0];
    FRONT_پڇاڙي_CHECK_IMPLEMENTED(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    خودڪار داخلا_0 = process_view_داخلا_new(context, 0);
    خودڪار داخلا_1 = process_view_داخلا_new(context, 1);
    ثابت خودڪار output_type = context.get_output_type();

    إذا (داخلا_0.get_element_type() != output_type) {
        داخلا_0 = std::make_shared<ov::op::v0::Convert>(داخلا_0, output_type);
    }
    إذا (داخلا_1.get_element_type() != output_type) {
        داخلا_1 = std::make_shared<ov::op::v0::Convert>(داخلا_1, output_type);
    }

    ثابت خودڪار axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    خودڪار res = std::make_shared<ov::op::v0::Concat>(Output_ويڪٽر{داخلا_0, داخلا_1}, axis);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
