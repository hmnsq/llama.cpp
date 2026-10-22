#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <cstring>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/mvn.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_norm(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);

    auto આગત_ગાંઠ = process_view_આગત_new(context, 0);
    દશાંશ eps;
    memcpy(&eps, context.get_output_op_params(), કદ(દશાંશ));

    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto res = std::make_shared<ov::op::v6::MVN>(આગત_ગાંઠ, axes, સાચું, eps, ov::op::MVNEpsMode::INSIDE_SQRT);

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
