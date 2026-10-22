#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/frontend/exception.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_concat(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 2, 2);

    ध्रुव int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CONCAT requires output op params");

    ध्रुव स्वतः output_shape = context.get_output_shape();
    FRONT_END_CHECK_IMPLEMENTED(output_shape.rank().is_static(), "CONCAT requires static output rank");

    ध्रुव स्वतः rank = output_shape.rank().get_length();
    ध्रुव int32_t ggml_dim = op_params[0];
    FRONT_END_CHECK_IMPLEMENTED(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    स्वतः निवेश_0 = process_view_निवेश_new(context, 0);
    स्वतः निवेश_1 = process_view_निवेश_new(context, 1);
    ध्रुव स्वतः output_type = context.get_output_type();

    यदि (निवेश_0.get_element_type() != output_type) {
        निवेश_0 = std::make_shared<ov::op::v0::Convert>(निवेश_0, output_type);
    }
    यदि (निवेश_1.get_element_type() != output_type) {
        निवेश_1 = std::make_shared<ov::op::v0::Convert>(निवेश_1, output_type);
    }

    ध्रुव स्वतः axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    स्वतः res = std::make_shared<ov::op::v0::Concat>(OutputVector{निवेश_0, निवेश_1}, axis);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
