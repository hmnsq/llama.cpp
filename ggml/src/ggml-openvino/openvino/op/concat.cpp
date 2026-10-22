#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/frontend/exception.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

Output_ಸದಿಶ translate_concat(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);

    ಸ್ಥಿರಾಂಕ int32_t * op_params = context.get_output_op_params();
    FRONT_ಕೊನೆ_CHECK_IMPLEMENTED(op_params != ಶೂನ್ಯಸೂಚಕ, "CONCAT requires output op params");

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ output_shape = context.get_output_shape();
    FRONT_ಕೊನೆ_CHECK_IMPLEMENTED(output_shape.rank().is_static(), "CONCAT requires static output rank");

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ rank = output_shape.rank().get_ಉದ್ದಳತೆ();
    ಸ್ಥಿರಾಂಕ int32_t ggml_dim = op_params[0];
    FRONT_ಕೊನೆ_CHECK_IMPLEMENTED(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_0 = process_view_ಒಳಹರಿವು_new(context, 0);
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_1 = process_view_ಒಳಹರಿವು_new(context, 1);
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ output_type = context.get_output_type();

    ಆದರೆ (ಒಳಹರಿವು_0.get_element_type() != output_type) {
        ಒಳಹರಿವು_0 = std::make_shared<ov::op::v0::Convert>(ಒಳಹರಿವು_0, output_type);
    }
    ಆದರೆ (ಒಳಹರಿವು_1.get_element_type() != output_type) {
        ಒಳಹರಿವು_1 = std::make_shared<ov::op::v0::Convert>(ಒಳಹರಿವು_1, output_type);
    }

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v0::Concat>(Output_ಸದಿಶ{ಒಳಹರಿವು_0, ಒಳಹರಿವು_1}, axis);

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
