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

Output_ویکټور translate_concat(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 2, 2);

    ثابت int32_t * op_params = context.get_output_op_params();
    FRONT_پای_ټکی_CHECK_IMPLEMENTED(op_params != تش_اشاره, "CONCAT requires output op params");

    ثابت خپلکاری output_shape = context.get_output_shape();
    FRONT_پای_ټکی_CHECK_IMPLEMENTED(output_shape.rank().is_static(), "CONCAT requires static output rank");

    ثابت خپلکاری rank = output_shape.rank().get_اوږدتیا();
    ثابت int32_t ggml_dim = op_params[0];
    FRONT_پای_ټکی_CHECK_IMPLEMENTED(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    خپلکاری ننوت_0 = process_view_ننوت_new(context, 0);
    خپلکاری ننوت_1 = process_view_ننوت_new(context, 1);
    ثابت خپلکاری output_type = context.get_output_type();

    إذا (ننوت_0.get_element_type() != output_type) {
        ننوت_0 = std::make_shared<ov::op::v0::Convert>(ننوت_0, output_type);
    }
    إذا (ننوت_1.get_element_type() != output_type) {
        ننوت_1 = std::make_shared<ov::op::v0::Convert>(ننوت_1, output_type);
    }

    ثابت خپلکاری axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    خپلکاری res = std::make_shared<ov::op::v0::Concat>(Output_ویکټور{ننوت_0, ننوت_1}, axis);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
