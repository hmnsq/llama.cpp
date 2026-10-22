#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <cstring>
#अन्तर्भाव <openvino/op/clamp.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_clamp(स्थिरांक गाठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    स्वतः आदान = process_view_आदान_new(context, 0);

    स्थिरांक int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CLAMP requires output op params");

    दशांश min;
    दशांश max;
    std::स_नकल(&min, reinterpret_cast<स्थिरांक दशांश *>(op_params) + 0, आकार(दशांश));
    std::स_नकल(&max, reinterpret_cast<स्थिरांक दशांश *>(op_params) + 1, आकार(दशांश));

    स्वतः res = std::make_shared<ov::op::v0::Clamp>(आदान, min, max);
    परत rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
