#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <cstdint>
#ضمن <openvino/frontend/exception.hpp>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/range.hpp>
#ضمن <openvino/op/reduce_prod.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/scatter_update.hpp>
#ضمن <openvino/op/shape_of.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

// GGML SET writes src1 into a view of src0 and returns the updated tensor.
Output_سمتیہ translate_سیٹ(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 2, 2);

    خودکار dst = process_view_داخلہ_new(context, 0);
    خودکار src = process_view_داخلہ_new(context, 1);

    src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());

    ثابت خودکار dst_stride = context.get_داخلہ_stride(0);
    FRONT_سرا_OP_CONVERSION_CHECK(dst_stride.size() >= 4, "SET requires 4D destination strides");

    ثابت خودکار * op_params = reinterpret_cast<ثابت uint32_t *>(context.get_output_op_params());
    ثابت size_t وتھ = static_cast<size_t>(op_params[3]);

    ثابت size_t elem_ناپ = dst_stride.back();
    FRONT_سرا_OP_CONVERSION_CHECK(elem_ناپ != 0 && وتھ % elem_ناپ == 0,
                                  "SET offset must be aligned to destination element size");

    ثابت int64_t وتھ_elems = static_cast<int64_t>(وتھ / elem_ناپ);

    خودکار dst_flat = std::make_shared<ov::op::v1::Reshape>(
        dst,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        خطأ);

    خودکار src_flat = std::make_shared<ov::op::v1::Reshape>(
        src,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        خطأ);

    خودکار src_shape = std::make_shared<ov::op::v3::Shape_دا>(src_flat, ov::element::i64);
    خودکار src_len = std::make_shared<ov::op::v1::ReduceProd>(
        src_shape,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
        خطأ);

    خودکار start = ov::op::v0::Constant::create(ov::element::i64, {}, {وتھ_elems});
    خودکار stop = std::make_shared<ov::op::v1::جوڑو>(start, src_len);
    خودکار قدم = ov::op::v0::Constant::create(ov::element::i64, {}, {1});

    خودکار indices = std::make_shared<ov::op::v4::Range>(start, stop, قدم, ov::element::i64);
    خودکار axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    خودکار updated_flat = std::make_shared<ov::op::v3::ScatterUpdate>(dst_flat, indices, src_flat, axis);

    خودکار dst_shape = std::make_shared<ov::op::v3::Shape_دا>(dst, ov::element::i64);
    خودکار res = std::make_shared<ov::op::v1::Reshape>(updated_flat, dst_shape, خطأ);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
