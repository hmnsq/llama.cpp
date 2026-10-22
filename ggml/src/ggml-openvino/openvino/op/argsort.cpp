#അന്തര്ഭാവ "../node_context.h"
#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"
#അന്തര്ഭാവ "ggml.h"

#അന്തര്ഭാവ <openvino/frontend/exception.hpp>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/squeeze.hpp>
#അന്തര്ഭാവ <openvino/op/topk.hpp>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

OutputVector translate_argsort(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);

    auto നിവേശം = process_view_നിവേശം_new(context, 0);

    സ്ഥിരം int32_t order = context.get_output_op_params()[0];

    ov::op::v11::മുകൾ_K::Mode mode;
    തിരഞ്ഞെടുപ്പ് (order) {
    അവസ്ഥ GGML_SORT_ORDER_ASC:
        mode = ov::op::v11::മുകൾ_K::Mode::MIN;
        നിർത്തുക;
    അവസ്ഥ GGML_SORT_ORDER_DESC:
        mode = ov::op::v11::മുകൾ_K::Mode::MAX;
        നിർത്തുക;
    സ്ഥിരസ്ഥിതി:
        FRONT_END_OP_CONVERSION_CHECK(അസത്യം, "Unsupported GGML_OP_ARGSORT order: ", order);
    }

    // Stateful models drop the leading size-1 batch dim, so the expert axis is 2 there
    // instead of 3 (same rank-3-vs-rank-4 split as get_rows.cpp / process_view_input).
    സ്ഥിരം പൂർണ്ണസംഖ്യ axis = (context.is_stateful() && നിവേശം.get_partial_shape().rank() == 3) ? 2 : 3;

    auto k = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(നിവേശം.get_കണ്ണി_shared_ptr(), {axis}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    auto topk = std::make_shared<ov::op::v11::മുകൾ_K>(നിവേശം, k, axis, mode, ov::op::v11::മുകൾ_K::SortType::SORT_മൂല്യങ്ങൾ,
                                                    context.get_output_type(), അസത്യം);

    തിരികെ rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
