#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/sqrt.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

Output_सदिशम् translate_sqr(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);

    स्वचालित निवेशः = process_view_निवेशः_new(context, 0);
    स्वचालित res = std::make_shared<ov::op::v1::Multiply>(निवेशः, निवेशः);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

Output_सदिशम् translate_sqrt(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);

    स्वचालित निवेशः = process_view_निवेशः_new(context, 0);
    स्वचालित res = std::make_shared<ov::op::v0::Sqrt>(निवेशः);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
