#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"

#অন্তর্ভাব <memory>
#অন্তর্ভাব <openvino/frontend/exception.hpp>
#অন্তর্ভাব <openvino/op/concat.hpp>
#অন্তর্ভাব <openvino/op/convert.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_concat(ধ্রুব গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 2, 2);

    ধ্রুব int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CONCAT requires output op params");

    ধ্রুব auto output_shape = context.get_output_shape();
    FRONT_END_CHECK_IMPLEMENTED(output_shape.rank().is_static(), "CONCAT requires static output rank");

    ধ্রুব auto rank = output_shape.rank().get_length();
    ধ্রুব int32_t ggml_dim = op_params[0];
    FRONT_END_CHECK_IMPLEMENTED(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    auto প্রবেশ_0 = process_view_প্রবেশ_new(context, 0);
    auto প্রবেশ_1 = process_view_প্রবেশ_new(context, 1);
    ধ্রুব auto output_type = context.get_output_type();

    যদি (প্রবেশ_0.get_element_type() != output_type) {
        প্রবেশ_0 = std::make_shared<ov::op::v0::Convert>(প্রবেশ_0, output_type);
    }
    যদি (প্রবেশ_1.get_element_type() != output_type) {
        প্রবেশ_1 = std::make_shared<ov::op::v0::Convert>(প্রবেশ_1, output_type);
    }

    ধ্রুব auto axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    auto res = std::make_shared<ov::op::v0::Concat>(OutputVector{প্রবেশ_0, প্রবেশ_1}, axis);

    প্রত্যাগম rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
