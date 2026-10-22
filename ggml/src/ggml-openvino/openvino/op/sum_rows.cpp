#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/reduce_sum.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

Output_सदिशम् translate_योग_पङ्क्तिकाः(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);

    स्वचालित निवेशः = process_view_निवेशः_new(context, 0);
    स्वचालित res = std::make_shared<ov::op::v1::Reduce_योग>(
        निवेशः, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), सत्य);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
