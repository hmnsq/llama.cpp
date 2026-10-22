#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <openvino/frontend/exception.hpp>
#અન્તર્ભાવ <openvino/op/concat.hpp>
#અન્તર્ભાવ <openvino/op/convert.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_concat(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 2, 2);

    અચલ int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CONCAT requires output op params");

    અચલ auto output_shape = context.get_output_shape();
    FRONT_END_CHECK_IMPLEMENTED(output_shape.rank().is_static(), "CONCAT requires static output rank");

    અચલ auto rank = output_shape.rank().get_length();
    અચલ int32_t ggml_dim = op_params[0];
    FRONT_END_CHECK_IMPLEMENTED(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    auto આગત_0 = process_view_આગત_new(context, 0);
    auto આગત_1 = process_view_આગત_new(context, 1);
    અચલ auto output_type = context.get_output_type();

    જો (આગત_0.get_element_type() != output_type) {
        આગત_0 = std::make_shared<ov::op::v0::Convert>(આગત_0, output_type);
    }
    જો (આગત_1.get_element_type() != output_type) {
        આગત_1 = std::make_shared<ov::op::v0::Convert>(આગત_1, output_type);
    }

    અચલ auto axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    auto res = std::make_shared<ov::op::v0::Concat>(OutputVector{આગત_0, આગત_1}, axis);

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
