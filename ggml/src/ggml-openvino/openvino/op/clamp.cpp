#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <openvino/op/clamp.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

Output_ಸದಿಶ translate_clamp(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);

    ಸ್ಥಿರಾಂಕ int32_t * op_params = context.get_output_op_params();
    FRONT_ಕೊನೆ_CHECK_IMPLEMENTED(op_params != ಶೂನ್ಯಸೂಚಕ, "CLAMP requires output op params");

    ದಶಮಾಂಶ min;
    ದಶಮಾಂಶ max;
    std::ಸ್ಮರಣೆ_ನಕಲು(&min, reinterpret_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(op_params) + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    std::ಸ್ಮರಣೆ_ನಕಲು(&max, reinterpret_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(op_params) + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v0::Clamp>(ಒಳಹರಿವು, min, max);
    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
