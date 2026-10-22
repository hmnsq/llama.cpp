#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <cstring>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/mvn.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_norm(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);

    auto निवेशः_ग्रन्थिः = process_view_निवेशः_new(context, 0);
    भिन्न eps;
    memcpy(&eps, context.get_output_op_params(), परिमाण(भिन्न));

    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto res = std::make_shared<ov::op::v6::MVN>(निवेशः_ग्रन्थिः, axes, सच्च, eps, ov::op::MVNEpsMode::INSIDE_SQRT);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
