#অন্তর্ভাৱ "../node_context.h"
#অন্তর্ভাৱ "../op_table.h"
#অন্তর্ভাৱ "../utils.h"

#অন্তর্ভাৱ <memory>
#অন্তর্ভাৱ <openvino/frontend/exception.hpp>
#অন্তর্ভাৱ <openvino/op/concat.hpp>
#অন্তর্ভাৱ <openvino/op/convert.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_concat(ধ্রুৱ গ্ৰন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 2, 2);

    ধ্রুৱ int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CONCAT requires output op params");

    ধ্রুৱ auto output_shape = context.get_output_shape();
    FRONT_END_CHECK_IMPLEMENTED(output_shape.rank().is_static(), "CONCAT requires static output rank");

    ধ্রুৱ auto rank = output_shape.rank().get_length();
    ধ্রুৱ int32_t ggml_dim = op_params[0];
    FRONT_END_CHECK_IMPLEMENTED(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    auto প্ৰবেশ_0 = process_view_প্ৰবেশ_new(context, 0);
    auto প্ৰবেশ_1 = process_view_প্ৰবেশ_new(context, 1);
    ধ্রুৱ auto output_type = context.get_output_type();

    যদি (প্ৰবেশ_0.get_element_type() != output_type) {
        প্ৰবেশ_0 = std::make_shared<ov::op::v0::Convert>(প্ৰবেশ_0, output_type);
    }
    যদি (প্ৰবেশ_1.get_element_type() != output_type) {
        প্ৰবেশ_1 = std::make_shared<ov::op::v0::Convert>(প্ৰবেশ_1, output_type);
    }

    ধ্রুৱ auto axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    auto res = std::make_shared<ov::op::v0::Concat>(OutputVector{প্ৰবেশ_0, প্ৰবেশ_1}, axis);

    ঘূৰাই rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
