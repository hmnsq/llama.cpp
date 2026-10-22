#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <cstring>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/mvn.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_norm(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);

    auto உள்ளீடு_முனை = process_view_உள்ளீடு_new(context, 0);
    தசம eps;
    memcpy(&eps, context.get_output_op_params(), அளவு(தசம));

    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto res = std::make_shared<ov::op::v6::MVN>(உள்ளீடு_முனை, axes, உண்மை, eps, ov::op::MVNEpsMode::INSIDE_SQRT);

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
