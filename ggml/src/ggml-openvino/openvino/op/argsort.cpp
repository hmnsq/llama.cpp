#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"
#अन्तर्भाव "ggml.h"

#अन्तर्भाव <openvino/frontend/exception.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/squeeze.hpp>
#अन्तर्भाव <openvino/op/topk.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_argsort(अचर गाँठो_Context & context) {
    अङ्क_inputs_check(context, 1, 1);

    स्वतः आगत = process_view_आगत_new(context, 0);

    अचर int32_t order = context.get_output_op_params()[0];

    ov::op::v11::शीर्ष_K::Mode mode;
    विकल्प (order) {
    पक्ष GGML_SORT_ORDER_ASC:
        mode = ov::op::v11::शीर्ष_K::Mode::MIN;
        रोक;
    पक्ष GGML_SORT_ORDER_DESC:
        mode = ov::op::v11::शीर्ष_K::Mode::MAX;
        रोक;
    स्वभाव:
        FRONT_END_OP_CONVERSION_CHECK(असत्य, "Unsupported GGML_OP_ARGSORT order: ", order);
    }

    // Stateful models drop the leading size-1 batch dim, so the expert axis is 2 there
    // instead of 3 (same rank-3-vs-rank-4 split as get_rows.cpp / process_view_input).
    अचर पूर्णाङ्क axis = (context.is_stateful() && आगत.get_partial_shape().rank() == 3) ? 2 : 3;

    स्वतः k = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(आगत.get_गाँठो_shared_ptr(), {axis}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    स्वतः topk = std::make_shared<ov::op::v11::शीर्ष_K>(आगत, k, axis, mode, ov::op::v11::शीर्ष_K::SortType::SORT_मानहरू,
                                                    context.get_output_type(), असत्य);

    फर्काऊ rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
