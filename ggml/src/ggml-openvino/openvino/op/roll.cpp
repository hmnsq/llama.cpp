#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <openvino/op/roll.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_roll(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);
    மாறிலி int32_t * params = context.get_output_op_params();

    int64_t s0 = params[0];
    int64_t s1 = params[1];
    int64_t s2 = params[2];
    int64_t s3 = params[3];

    auto உள்ளீடு = context.get_உள்ளீடு(0);

    auto shift = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4}, std::vector<int64_t>{s3, s2, s1, s0});
    auto axes  = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 1, 2, 3});

    auto roll = std::make_shared<ov::op::v7::Roll>(உள்ளீடு, shift, axes);
    திரும்பு rename_outputs_with_suffix({roll}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
