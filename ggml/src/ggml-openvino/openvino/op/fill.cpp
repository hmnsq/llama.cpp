#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <openvino/op/broadcast.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

// GGML FILL sets all elements of a tensor to a constant value.
// The constant is stored as a float in op_params[0].
Output_ಸದಿಶ translate_fill(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    ದಶಮಾಂಶ c;
    ಸ್ಮರಣೆ_ನಕಲು(&c, context.get_output_op_params(), ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ವಯಂಚಾಲಿತ shape = context.get_ಒಳಹರಿವು_shape(0).ಗಮ್ಯ_shape();

    ಸ್ವಯಂಚಾಲಿತ val = ov::op::v0::Constant::create(context.get_output_type(), {}, {c});
    ಸ್ವಯಂಚಾಲಿತ ಗುರಿ_shape = ov::op::v0::Constant::create(ov::element::i64, {shape.size()},
        std::vector<int64_t>(shape.begin(), shape.end()));
    ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v3::ಪ್ರಸಾರ>(val, ಗುರಿ_shape);

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
