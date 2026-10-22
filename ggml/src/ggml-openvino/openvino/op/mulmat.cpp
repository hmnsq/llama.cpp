#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <climits>
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <openvino/core/node.hpp>
#અન્તર્ભાવ <openvino/core/node_output.hpp>
#અન્તર્ભાવ <openvino/op/broadcast.hpp>
#અન્તર્ભાવ <openvino/op/concat.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/convert.hpp>
#અન્તર્ભાવ <openvino/op/matmul.hpp>
#અન્તર્ભાવ <openvino/op/reshape.hpp>
#અન્તર્ભાવ <openvino/op/slice.hpp>
#અન્તર્ભાવ <openvino/op/transpose.hpp>
#અન્તર્ભાવ <openvino/op/unsqueeze.hpp>
#અન્તર્ભાવ <openvino/op/util/op_types.hpp>
#અન્તર્ભાવ <vector>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_mulmat(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 2, 2);

    પૂર્ણાંક op_case = context.get_op_case();

    ov::Output<ov::Node> B = process_view_આગત_new(context, 0);
    ov::Output<ov::Node> A = process_view_આગત_new(context, 1);

    જો (A.get_element_type() != B.get_element_type()) {
        B = std::make_shared<ov::op::v0::Convert>(B, context.get_આગત_type(1));
    }

    auto B_shape = context.get_આગત_shape(0).to_shape();
    auto A_shape = context.get_આગત_shape(1).to_shape();
    int64_t A_batch = A_shape[1];
    int64_t B_batch = B_shape[1];

    auto A_batch_larger = A_batch > B_batch;
    auto batch_large = A_batch_larger ? A_batch : B_batch;
    auto batch_small = A_batch_larger ? B_batch : A_batch;

    Output<Node> Z = A_batch_larger ? B : A;
    auto Z_shape = A_batch_larger ? B_shape : A_shape;
    int64_t factor = batch_large / batch_small;
    જો (factor > 1 && batch_small > 1) {
        auto batch_large_ગાંઠ = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_large});
        auto batch_small_ગાંઠ = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_small});
        auto factor_ગાંઠ = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{factor});

        auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
        auto Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

        auto broadcast_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
        auto new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                        {(int64_t) 0, batch_large, (int64_t) -1, (int64_t) Z_shape[3]});
        જો (op_case == 2) {
            new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                       {(int64_t) 0, batch_large, (int64_t) Z_shape[2], (int64_t) -1});
        }

        auto Z_broadcasted = std::make_shared<ov::op::v3::Broadcast>(Z_unsqueezed, broadcast_shape,
                                                                     ov::op::BroadcastType::BIDIRECTIONAL);
        Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_Z_shape, સાચું);
    }
    જો (A_batch_larger) {
        B = Z;
    } નહીંતર {
        A = Z;
    }

    તાર્કિક transpose_b = સાચું;
    ov::Output<Node> res = std::make_shared<ov::op::v0::MatMul>(A, B, ખોટું, transpose_b);

    અચલ auto output_type = context.get_output_type();
    જો (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
