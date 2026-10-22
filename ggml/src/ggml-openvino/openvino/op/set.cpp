#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <cstdint>
#شامل <openvino/frontend/exception.hpp>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/range.hpp>
#شامل <openvino/op/reduce_prod.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/scatter_update.hpp>
#شامل <openvino/op/shape_of.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

// GGML SET writes src1 into a view of src0 and returns the updated tensor.
Output_سمتیہ translate_سیٹ(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 2, 2);

    خودکار dst = process_view_داخلہ_new(context, 0);
    خودکار src = process_view_داخلہ_new(context, 1);

    src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());

    ثابت خودکار dst_stride = context.get_داخلہ_stride(0);
    FRONT_انتہا_OP_CONVERSION_CHECK(dst_stride.size() >= 4, "SET requires 4D destination strides");

    ثابت خودکار * op_params = reinterpret_cast<ثابت uint32_t *>(context.get_output_op_params());
    ثابت size_t انحراف = static_cast<size_t>(op_params[3]);

    ثابت size_t elem_سائز = dst_stride.back();
    FRONT_انتہا_OP_CONVERSION_CHECK(elem_سائز != 0 && انحراف % elem_سائز == 0,
                                  "SET offset must be aligned to destination element size");

    ثابت int64_t انحراف_elems = static_cast<int64_t>(انحراف / elem_سائز);

    خودکار dst_flat = std::make_shared<ov::op::v1::Reshape>(
        dst,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        جھوٹ);

    خودکار src_flat = std::make_shared<ov::op::v1::Reshape>(
        src,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        جھوٹ);

    خودکار src_shape = std::make_shared<ov::op::v3::Shape_کا>(src_flat, ov::element::i64);
    خودکار src_len = std::make_shared<ov::op::v1::ReduceProd>(
        src_shape,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
        جھوٹ);

    خودکار start = ov::op::v0::Constant::create(ov::element::i64, {}, {انحراف_elems});
    خودکار stop = std::make_shared<ov::op::v1::جوڑو>(start, src_len);
    خودکار قدم = ov::op::v0::Constant::create(ov::element::i64, {}, {1});

    خودکار indices = std::make_shared<ov::op::v4::Range>(start, stop, قدم, ov::element::i64);
    خودکار axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    خودکار updated_flat = std::make_shared<ov::op::v3::ScatterUpdate>(dst_flat, indices, src_flat, axis);

    خودکار dst_shape = std::make_shared<ov::op::v3::Shape_کا>(dst, ov::element::i64);
    خودکار res = std::make_shared<ov::op::v1::Reshape>(updated_flat, dst_shape, جھوٹ);

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
