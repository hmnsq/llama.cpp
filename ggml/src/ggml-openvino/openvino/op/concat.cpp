#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/frontend/exception.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/concat.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_concat(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 2, 2);

    ਅਚਲ int32_t * op_params = context.get_output_op_params();
    FRONT_ਸਿਰਾ_CHECK_IMPLEMENTED(op_params != ਖਾਲੀ_ਸੂਚਕ, "CONCAT requires output op params");

    ਅਚਲ ਸਵੈਚਾਲਿਤ output_shape = context.get_output_shape();
    FRONT_ਸਿਰਾ_CHECK_IMPLEMENTED(output_shape.rank().is_static(), "CONCAT requires static output rank");

    ਅਚਲ ਸਵੈਚਾਲਿਤ rank = output_shape.rank().get_ਲੰਮਾਈ();
    ਅਚਲ int32_t ggml_dim = op_params[0];
    FRONT_ਸਿਰਾ_CHECK_IMPLEMENTED(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_0 = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_1 = process_view_ਨਿਵੇਸ਼_new(context, 1);
    ਅਚਲ ਸਵੈਚਾਲਿਤ output_type = context.get_output_type();

    ਜੇ (ਨਿਵੇਸ਼_0.get_element_type() != output_type) {
        ਨਿਵੇਸ਼_0 = std::make_shared<ov::op::v0::Convert>(ਨਿਵੇਸ਼_0, output_type);
    }
    ਜੇ (ਨਿਵੇਸ਼_1.get_element_type() != output_type) {
        ਨਿਵੇਸ਼_1 = std::make_shared<ov::op::v0::Convert>(ਨਿਵੇਸ਼_1, output_type);
    }

    ਅਚਲ ਸਵੈਚਾਲਿਤ axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v0::Concat>(Output_ਸਦਿਸ਼{ਨਿਵੇਸ਼_0, ਨਿਵੇਸ਼_1}, axis);

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
