#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <cstring>
#अन्तर्भाव <openvino/op/clamp.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

Output_सदिशम् translate_clamp(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);

    स्वचालित निवेशः = process_view_निवेशः_new(context, 0);

    ध्रुव int32_t * op_params = context.get_output_op_params();
    FRONT_अन्तः_CHECK_IMPLEMENTED(op_params != शून्यसूचकः, "CLAMP requires output op params");

    भिन्न min;
    भिन्न max;
    std::स्मृतिं_प्रतिलिख(&min, reinterpret_cast<ध्रुव भिन्न *>(op_params) + 0, परिमाण(भिन्न));
    std::स्मृतिं_प्रतिलिख(&max, reinterpret_cast<ध्रुव भिन्न *>(op_params) + 1, परिमाण(भिन्न));

    स्वचालित res = std::make_shared<ov::op::v0::Clamp>(निवेशः, min, max);
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
