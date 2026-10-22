#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <cstring>
#अन्तर्भाव <openvino/op/clamp.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_clamp(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    स्वतः निवेश = process_view_निवेश_new(context, 0);

    ध्रुव int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CLAMP requires output op params");

    भिन्न min;
    भिन्न max;
    std::स_नकल(&min, reinterpret_cast<ध्रुव भिन्न *>(op_params) + 0, परिमाण(भिन्न));
    std::स_नकल(&max, reinterpret_cast<ध्रुव भिन्न *>(op_params) + 1, परिमाण(भिन्न));

    स्वतः res = std::make_shared<ov::op::v0::Clamp>(निवेश, min, max);
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
