#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <climits>
#समावेश <cstdint>
#समावेश <memory>
#समावेश <openvino/core/node.hpp>
#समावेश <openvino/core/node_output.hpp>
#समावेश <openvino/op/broadcast.hpp>
#समावेश <openvino/op/concat.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/convert.hpp>
#समावेश <openvino/op/matmul.hpp>
#समावेश <openvino/op/reshape.hpp>
#समावेश <openvino/op/slice.hpp>
#समावेश <openvino/op/transpose.hpp>
#समावेश <openvino/op/unsqueeze.hpp>
#समावेश <openvino/op/util/op_types.hpp>
#समावेश <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_mulmat(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 2, 2);

    पूर्णांक op_case = context.get_op_case();

    ov::Output<ov::Node> B = process_view_निवेश_new(context, 0);
    ov::Output<ov::Node> A = process_view_निवेश_new(context, 1);

    अगर (A.get_element_type() != B.get_element_type()) {
        B = std::make_shared<ov::op::v0::Convert>(B, context.get_निवेश_type(1));
    }

    स्वतः B_shape = context.get_निवेश_shape(0).to_shape();
    स्वतः A_shape = context.get_निवेश_shape(1).to_shape();
    int64_t A_batch = A_shape[1];
    int64_t B_batch = B_shape[1];

    स्वतः A_batch_larger = A_batch > B_batch;
    स्वतः batch_large = A_batch_larger ? A_batch : B_batch;
    स्वतः batch_small = A_batch_larger ? B_batch : A_batch;

    Output<Node> Z = A_batch_larger ? B : A;
    स्वतः Z_shape = A_batch_larger ? B_shape : A_shape;
    int64_t factor = batch_large / batch_small;
    अगर (factor > 1 && batch_small > 1) {
        स्वतः batch_large_गाँठ = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_large});
        स्वतः batch_small_गाँठ = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_small});
        स्वतः factor_गाँठ = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{factor});

        स्वतः unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
        स्वतः Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

        स्वतः broadcast_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
        स्वतः new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                        {(int64_t) 0, batch_large, (int64_t) -1, (int64_t) Z_shape[3]});
        अगर (op_case == 2) {
            new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                       {(int64_t) 0, batch_large, (int64_t) Z_shape[2], (int64_t) -1});
        }

        स्वतः Z_broadcasted = std::make_shared<ov::op::v3::Broadcast>(Z_unsqueezed, broadcast_shape,
                                                                     ov::op::BroadcastType::BIDIRECTIONAL);
        Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_Z_shape, true);
    }
    अगर (A_batch_larger) {
        B = Z;
    } अन्यथा {
        A = Z;
    }

    bool transpose_b = true;
    ov::Output<Node> res = std::make_shared<ov::op::v0::MatMul>(A, B, false, transpose_b);

    स्थिर स्वतः output_type = context.get_output_type();
    अगर (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
