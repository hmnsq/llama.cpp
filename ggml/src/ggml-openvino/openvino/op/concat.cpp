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

OutputVector translate_concat(अचर गाँठो_Context & context) {
    अङ्क_inputs_check(context, 2, 2);

    अचर int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CONCAT requires output op params");

    अचर स्वतः output_shape = context.get_output_shape();
    FRONT_END_CHECK_IMPLEMENTED(output_shape.rank().is_static(), "CONCAT requires static output rank");

    अचर स्वतः rank = output_shape.rank().get_length();
    अचर int32_t ggml_dim = op_params[0];
    FRONT_END_CHECK_IMPLEMENTED(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    स्वतः आगत_0 = process_view_आगत_new(context, 0);
    स्वतः आगत_1 = process_view_आगत_new(context, 1);
    अचर स्वतः output_type = context.get_output_type();

    यदि (आगत_0.get_element_type() != output_type) {
        आगत_0 = std::make_shared<ov::op::v0::Convert>(आगत_0, output_type);
    }
    यदि (आगत_1.get_element_type() != output_type) {
        आगत_1 = std::make_shared<ov::op::v0::Convert>(आगत_1, output_type);
    }

    अचर स्वतः axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    स्वतः res = std::make_shared<ov::op::v0::Concat>(OutputVector{आगत_0, आगत_1}, axis);

    फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
