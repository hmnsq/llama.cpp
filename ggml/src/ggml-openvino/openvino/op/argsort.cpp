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

OutputVector translate_argsort(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    auto ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);

    ಧ್ರುವ int32_t order = context.get_output_op_params()[0];

    ov::op::v11::ತುದಿ_K::Mode mode;
    ವಿಕಲ್ಪ (order) {
    ಪಕ್ಷ GGML_SORT_ORDER_ASC:
        mode = ov::op::v11::ತುದಿ_K::Mode::MIN;
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_SORT_ORDER_DESC:
        mode = ov::op::v11::ತುದಿ_K::Mode::MAX;
        ಭಙ್ಗ;
    ಸ್ವಭಾವ:
        FRONT_END_OP_CONVERSION_CHECK(ಅಸತ್ಯ, "Unsupported GGML_OP_ARGSORT order: ", order);
    }

    // Stateful models drop the leading size-1 batch dim, so the expert axis is 2 there
    // instead of 3 (same rank-3-vs-rank-4 split as get_rows.cpp / process_view_input).
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ axis = (context.is_stateful() && ಒಳಹರಿವು.get_partial_shape().rank() == 3) ? 2 : 3;

    auto k = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(ಒಳಹರಿವು.get_ಗಂಟು_shared_ptr(), {axis}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    auto topk = std::make_shared<ov::op::v11::ತುದಿ_K>(ಒಳಹರಿವು, k, axis, mode, ov::op::v11::ತುದಿ_K::SortType::SORT_ಮೌಲ್ಯಗಳು,
                                                    context.get_output_type(), ಅಸತ್ಯ);

    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
