#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <climits>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/core/node.hpp>
#అన్తర్భావ <openvino/core/node_output.hpp>
#అన్తర్భావ <openvino/op/broadcast.hpp>
#అన్తర్భావ <openvino/op/concat.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/matmul.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/slice.hpp>
#అన్తర్భావ <openvino/op/transpose.hpp>
#అన్తర్భావ <openvino/op/unsqueeze.hpp>
#అన్తర్భావ <openvino/op/util/op_types.hpp>
#అన్తర్భావ <vector>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_mulmat(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 2, 2);

    పూర్ణాంకం op_case = context.get_op_case();

    ov::Output<ov::Node> B = process_view_ఆదానం_new(context, 0);
    ov::Output<ov::Node> A = process_view_ఆదానం_new(context, 1);

    ఒకవేళ (A.get_element_type() != B.get_element_type()) {
        B = std::make_shared<ov::op::v0::Convert>(B, context.get_ఆదానం_type(1));
    }

    స్వయంచాలక B_shape = context.get_ఆదానం_shape(0).గమ్యం_shape();
    స్వయంచాలక A_shape = context.get_ఆదానం_shape(1).గమ్యం_shape();
    int64_t A_batch = A_shape[1];
    int64_t B_batch = B_shape[1];

    స్వయంచాలక A_batch_larger = A_batch > B_batch;
    స్వయంచాలక batch_large = A_batch_larger ? A_batch : B_batch;
    స్వయంచాలక batch_small = A_batch_larger ? B_batch : A_batch;

    Output<Node> Z = A_batch_larger ? B : A;
    స్వయంచాలక Z_shape = A_batch_larger ? B_shape : A_shape;
    int64_t గుణకం = batch_large / batch_small;
    ఒకవేళ (గుణకం > 1 && batch_small > 1) {
        స్వయంచాలక batch_large_కణుపు = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_large});
        స్వయంచాలక batch_small_కణుపు = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_small});
        స్వయంచాలక గుణకం_కణుపు = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{గుణకం});

        స్వయంచాలక unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
        స్వయంచాలక Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

        స్వయంచాలక ప్రసారం_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, గుణకం, (int64_t) 1, (int64_t) 1});
        స్వయంచాలక new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                        {(int64_t) 0, batch_large, (int64_t) -1, (int64_t) Z_shape[3]});
        ఒకవేళ (op_case == 2) {
            new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                       {(int64_t) 0, batch_large, (int64_t) Z_shape[2], (int64_t) -1});
        }

        స్వయంచాలక Z_broadcasted = std::make_shared<ov::op::v3::ప్రసారం>(Z_unsqueezed, ప్రసారం_shape,
                                                                     ov::op::ప్రసారం_Type::BIDIRECTIONAL);
        Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_Z_shape, నిజం);
    }
    ఒకవేళ (A_batch_larger) {
        B = Z;
    } లేకపోతే {
        A = Z;
    }

    తార్కిక transpose_b = నిజం;
    ov::Output<Node> res = std::make_shared<ov::op::v0::MatMul>(A, B, అబద్ధం, transpose_b);

    స్థిరరాశి స్వయంచాలక output_type = context.get_output_type();
    ఒకవేళ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
