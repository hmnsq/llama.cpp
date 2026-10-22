#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"
#समावेश "ggml.h"

#समावेश <openvino/frontend/exception.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/squeeze.hpp>
#समावेश <openvino/op/topk.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_argsort(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    स्वतः निवेश = process_view_निवेश_new(context, 0);

    स्थिर int32_t order = context.get_output_op_params()[0];

    ov::op::v11::शीर्ष_K::Mode mode;
    चयन (order) {
    हाल GGML_SORT_ORDER_ASC:
        mode = ov::op::v11::शीर्ष_K::Mode::MIN;
        अवरोध;
    हाल GGML_SORT_ORDER_DESC:
        mode = ov::op::v11::शीर्ष_K::Mode::MAX;
        अवरोध;
    शेष:
        FRONT_END_OP_CONVERSION_CHECK(false, "Unsupported GGML_OP_ARGSORT order: ", order);
    }

    // Stateful models drop the leading size-1 batch dim, so the expert axis is 2 there
    // instead of 3 (same rank-3-vs-rank-4 split as get_rows.cpp / process_view_input).
    स्थिर पूर्णांक axis = (context.is_stateful() && निवेश.get_partial_shape().rank() == 3) ? 2 : 3;

    स्वतः k = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(निवेश.get_गाँठ_shared_ptr(), {axis}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    स्वतः topk = std::make_shared<ov::op::v11::शीर्ष_K>(निवेश, k, axis, mode, ov::op::v11::शीर्ष_K::SortType::SORT_मूल्यावली,
                                                    context.get_output_type(), false);

    वापस rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
