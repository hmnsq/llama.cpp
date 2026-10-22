#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"

#ಅನ್ತರ್ಭಾವ <openvino/frontend/exception.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/squeeze.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/topk.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

Output_ಸದಿಶ translate_argsort(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);

    ಸ್ಥಿರಾಂಕ int32_t ಕ್ರಮ = context.get_output_op_params()[0];

    ov::op::v11::ತುದಿ_K::ಬಹುಲಕ_2 mode;
    ಆಯ್ಕೆ (ಕ್ರಮ) {
    ಸ್ಥಿತಿ GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC:
        mode = ov::op::v11::ತುದಿ_K::ಬಹುಲಕ_2::MIN;
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC:
        mode = ov::op::v11::ತುದಿ_K::ಬಹುಲಕ_2::MAX;
        ನಿಲ್ಲಿಸು;
    ಪೂರ್ವನಿಯೋಜಿತ:
        FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(ಸುಳ್ಳು, "Unsupported GGML_OP_ARGSORT order: ", ಕ್ರಮ);
    }

    // Stateful models drop the leading size-1 batch dim, so the expert axis is 2 there
    // instead of 3 (same rank-3-vs-rank-4 split as get_rows.cpp / process_view_input).
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ axis = (context.is_stateful() && ಒಳಹರಿವು.get_partial_shape().rank() == 3) ? 2 : 3;

    ಸ್ವಯಂಚಾಲಿತ k = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(ಒಳಹರಿವು.get_ಗಂಟು_shared_ptr(), {axis}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    ಸ್ವಯಂಚಾಲಿತ topk = std::make_shared<ov::op::v11::ತುದಿ_K>(ಒಳಹರಿವು, k, axis, mode, ov::op::v11::ತುದಿ_K::ವಿಂಗಡಿಸು_Type::ವಿಂಗಡಿಸು_ಮೌಲ್ಯಗಳು,
                                                    context.get_output_type(), ಸುಳ್ಳು);

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
