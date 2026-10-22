#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/reshape.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_reshape(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    ᯀᯮᯝ (context.get_input(0).get_partial_shape().is_static() &&
        context.get_input_shape(0) == context.get_output_shape()) {
        ᯀᯮᯞ {context.get_input(0)};
    }

    const puur_naa_nka op_case = context.get_op_case();
    const auto output_shape = context.get_output_shape().to_shape();
    std::vector<int64_t> shape(output_shape.begin(), output_shape.end());
    std::shared_ptr<ov::Node> new_shape_node;
    switch (op_case) {
    pak_sa 0:
        bha_nga;
    pak_sa 1:
    pak_sa 9:
        shape[1] = -1;
        ᯀᯮᯝ (context.is_stateful() && op_case == 1) {
            shape.erase(shape.begin());
        }
        bha_nga;
    pak_sa 2:
    pak_sa 3:
        shape[2] = -1;
        ᯀᯮᯝ (op_case == 3) {
            shape[3] = 1;
        }
        bha_nga;
    pak_sa 4:
        ᯀᯮᯞ {context.get_input(0).get_node_shared_ptr()->input_value(0)};
    pak_sa 5:
    pak_sa 7:
        shape = {1, 1, -1, shape[3]};
        ᯀᯮᯝ (context.is_stateful() && op_case == 5) {
            shape.erase(shape.begin());
        }
        bha_nga;
    pak_sa 6:
        // Recurrent inputs keep the active sequence count separate from the token count.
        ᯀᯮᯝ (context.has_input("s_copy_active_slot_len")) {
            auto n_slot_active_len = context.get_input("s_copy_active_slot_len");
            auto emb_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {shape[3]});
            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            new_shape_node =
                std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one, n_slot_active_len, neg_one, emb_size}, 0);
        } ᯀᯮᯝᯂᯪ {
            shape = {1, 1, -1, shape[3]};
        }
        bha_nga;
    pak_sa 8:
        shape[0] = -1;
        bha_nga;
    default:
        FRONT_END_OP_CONVERSION_CHECK(false, "Unsupported RESHAPE case: ", op_case);
    }
    ᯀᯮᯝ (!new_shape_node) {
        new_shape_node = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
    }
    auto res = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), new_shape_node, false);
    ᯀᯮᯞ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
