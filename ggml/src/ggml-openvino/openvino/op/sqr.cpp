#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"

#অন্তর্ভাব <memory>
#অন্তর্ভাব <openvino/op/multiply.hpp>
#অন্তর্ভাব <openvino/op/sqrt.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_sqr(ধ্রুব গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    auto প্রবেশ = process_view_প্রবেশ_new(context, 0);
    auto res = std::make_shared<ov::op::v1::Multiply>(প্রবেশ, প্রবেশ);

    প্রত্যাগম rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_sqrt(ধ্রুব গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    auto প্রবেশ = process_view_প্রবেশ_new(context, 0);
    auto res = std::make_shared<ov::op::v0::Sqrt>(প্রবেশ);

    প্রত্যাগম rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
