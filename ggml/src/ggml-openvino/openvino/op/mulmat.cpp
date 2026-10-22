#അന്തര്ഭാവ "../node_context.h"
#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"

#അന്തര്ഭാവ <climits>
#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <openvino/core/node.hpp>
#അന്തര്ഭാവ <openvino/core/node_output.hpp>
#അന്തര്ഭാവ <openvino/op/broadcast.hpp>
#അന്തര്ഭാവ <openvino/op/concat.hpp>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/convert.hpp>
#അന്തര്ഭാവ <openvino/op/matmul.hpp>
#അന്തര്ഭാവ <openvino/op/reshape.hpp>
#അന്തര്ഭാവ <openvino/op/slice.hpp>
#അന്തര്ഭാവ <openvino/op/transpose.hpp>
#അന്തര്ഭാവ <openvino/op/unsqueeze.hpp>
#അന്തര്ഭാവ <openvino/op/util/op_types.hpp>
#അന്തര്ഭാവ <vector>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

OutputVector translate_mulmat(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 2, 2);

    പൂർണ്ണസംഖ്യ op_case = context.get_op_case();

    ov::Output<ov::Node> B = process_view_നിവേശം_new(context, 0);
    ov::Output<ov::Node> A = process_view_നിവേശം_new(context, 1);

    എങ്കിൽ (A.get_element_type() != B.get_element_type()) {
        B = std::make_shared<ov::op::v0::Convert>(B, context.get_നിവേശം_type(1));
    }

    auto B_shape = context.get_നിവേശം_shape(0).to_shape();
    auto A_shape = context.get_നിവേശം_shape(1).to_shape();
    int64_t A_batch = A_shape[1];
    int64_t B_batch = B_shape[1];

    auto A_batch_larger = A_batch > B_batch;
    auto batch_large = A_batch_larger ? A_batch : B_batch;
    auto batch_small = A_batch_larger ? B_batch : A_batch;

    Output<Node> Z = A_batch_larger ? B : A;
    auto Z_shape = A_batch_larger ? B_shape : A_shape;
    int64_t factor = batch_large / batch_small;
    എങ്കിൽ (factor > 1 && batch_small > 1) {
        auto batch_large_കണ്ണി = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_large});
        auto batch_small_കണ്ണി = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_small});
        auto factor_കണ്ണി = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{factor});

        auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
        auto Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

        auto broadcast_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
        auto new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                        {(int64_t) 0, batch_large, (int64_t) -1, (int64_t) Z_shape[3]});
        എങ്കിൽ (op_case == 2) {
            new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                       {(int64_t) 0, batch_large, (int64_t) Z_shape[2], (int64_t) -1});
        }

        auto Z_broadcasted = std::make_shared<ov::op::v3::Broadcast>(Z_unsqueezed, broadcast_shape,
                                                                     ov::op::BroadcastType::BIDIRECTIONAL);
        Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_Z_shape, സത്യം);
    }
    എങ്കിൽ (A_batch_larger) {
        B = Z;
    } അല്ലെങ്കിൽ {
        A = Z;
    }

    തർക്കം transpose_b = സത്യം;
    ov::Output<Node> res = std::make_shared<ov::op::v0::MatMul>(A, B, അസത്യം, transpose_b);

    സ്ഥിരം auto output_type = context.get_output_type();
    എങ്കിൽ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
