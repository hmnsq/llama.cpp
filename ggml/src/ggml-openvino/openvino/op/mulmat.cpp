#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <climits>
#ضمن <cstdint>
#ضمن <memory>
#ضمن <openvino/core/node.hpp>
#ضمن <openvino/core/node_output.hpp>
#ضمن <openvino/op/broadcast.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/matmul.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <openvino/op/unsqueeze.hpp>
#ضمن <openvino/op/util/op_types.hpp>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_ویکټور translate_mulmat(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 2, 2);

    صحيح op_case = context.get_op_case();

    ov::Output<ov::Node> B = process_view_ننوت_new(context, 0);
    ov::Output<ov::Node> A = process_view_ننوت_new(context, 1);

    إذا (A.get_element_type() != B.get_element_type()) {
        B = std::make_shared<ov::op::v0::Convert>(B, context.get_ننوت_type(1));
    }

    خپلکاری B_shape = context.get_ننوت_shape(0).منزل_shape();
    خپلکاری A_shape = context.get_ننوت_shape(1).منزل_shape();
    int64_t A_batch = A_shape[1];
    int64_t B_batch = B_shape[1];

    خپلکاری A_batch_larger = A_batch > B_batch;
    خپلکاری batch_large = A_batch_larger ? A_batch : B_batch;
    خپلکاری batch_small = A_batch_larger ? B_batch : A_batch;

    Output<Node> Z = A_batch_larger ? B : A;
    خپلکاری Z_shape = A_batch_larger ? B_shape : A_shape;
    int64_t ضریب = batch_large / batch_small;
    إذا (ضریب > 1 && batch_small > 1) {
        خپلکاری batch_large_غوټه = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_large});
        خپلکاری batch_small_غوټه = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_small});
        خپلکاری ضریب_غوټه = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{ضریب});

        خپلکاری unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
        خپلکاری Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

        خپلکاری خپرونه_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, ضریب, (int64_t) 1, (int64_t) 1});
        خپلکاری new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                        {(int64_t) 0, batch_large, (int64_t) -1, (int64_t) Z_shape[3]});
        إذا (op_case == 2) {
            new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                       {(int64_t) 0, batch_large, (int64_t) Z_shape[2], (int64_t) -1});
        }

        خپلکاری Z_broadcasted = std::make_shared<ov::op::v3::خپرونه>(Z_unsqueezed, خپرونه_shape,
                                                                     ov::op::خپرونه_Type::BIDIRECTIONAL);
        Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_Z_shape, ریښتیا);
    }
    إذا (A_batch_larger) {
        B = Z;
    } وإلا {
        A = Z;
    }

    منطقي transpose_b = ریښتیا;
    ov::Output<Node> res = std::make_shared<ov::op::v0::MatMul>(A, B, خطأ, transpose_b);

    ثابت خپلکاری output_type = context.get_output_type();
    إذا (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
