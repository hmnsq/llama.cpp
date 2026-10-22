#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"

#অন্তর্ভাব <cstring>
#অন্তর্ভাব <memory>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/mvn.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_norm(ধ্রুব গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    auto প্রবেশ_গ্রন্থি = process_view_প্রবেশ_new(context, 0);
    ভিন্ন eps;
    memcpy(&eps, context.get_output_op_params(), পরিমাণ(ভিন্ন));

    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto res = std::make_shared<ov::op::v6::MVN>(প্রবেশ_গ্রন্থি, axes, সত্য, eps, ov::op::MVNEpsMode::INSIDE_SQRT);

    প্রত্যাগম rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
