#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <openvino/frontend/exception.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/range.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reduce_prod.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/scatter_update.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/shape_of.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

// GGML SET writes src1 into a view of src0 and returns the updated tensor.
Output_ಸದಿಶ translate_ಗಣ(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);

    ಸ್ವಯಂಚಾಲಿತ dst = process_view_ಒಳಹರಿವು_new(context, 0);
    ಸ್ವಯಂಚಾಲಿತ src = process_view_ಒಳಹರಿವು_new(context, 1);

    src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ dst_stride = context.get_ಒಳಹರಿವು_stride(0);
    FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(dst_stride.size() >= 4, "SET requires 4D destination strides");

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * op_params = reinterpret_cast<ಸ್ಥಿರಾಂಕ uint32_t *>(context.get_output_op_params());
    ಸ್ಥಿರಾಂಕ size_t ಸ್ಥಾನಾಂತರ = static_cast<size_t>(op_params[3]);

    ಸ್ಥಿರಾಂಕ size_t elem_ಗಾತ್ರ = dst_stride.back();
    FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(elem_ಗಾತ್ರ != 0 && ಸ್ಥಾನಾಂತರ % elem_ಗಾತ್ರ == 0,
                                  "SET offset must be aligned to destination element size");

    ಸ್ಥಿರಾಂಕ int64_t ಸ್ಥಾನಾಂತರ_elems = static_cast<int64_t>(ಸ್ಥಾನಾಂತರ / elem_ಗಾತ್ರ);

    ಸ್ವಯಂಚಾಲಿತ dst_flat = std::make_shared<ov::op::v1::Reshape>(
        dst,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        ಸುಳ್ಳು);

    ಸ್ವಯಂಚಾಲಿತ src_flat = std::make_shared<ov::op::v1::Reshape>(
        src,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        ಸುಳ್ಳು);

    ಸ್ವಯಂಚಾಲಿತ src_shape = std::make_shared<ov::op::v3::Shape_ದ>(src_flat, ov::element::i64);
    ಸ್ವಯಂಚಾಲಿತ src_len = std::make_shared<ov::op::v1::ReduceProd>(
        src_shape,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
        ಸುಳ್ಳು);

    ಸ್ವಯಂಚಾಲಿತ start = ov::op::v0::Constant::create(ov::element::i64, {}, {ಸ್ಥಾನಾಂತರ_elems});
    ಸ್ವಯಂಚಾಲಿತ stop = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(start, src_len);
    ಸ್ವಯಂಚಾಲಿತ ಹೆಜ್ಜೆ = ov::op::v0::Constant::create(ov::element::i64, {}, {1});

    ಸ್ವಯಂಚಾಲಿತ indices = std::make_shared<ov::op::v4::Range>(start, stop, ಹೆಜ್ಜೆ, ov::element::i64);
    ಸ್ವಯಂಚಾಲಿತ axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    ಸ್ವಯಂಚಾಲಿತ updated_flat = std::make_shared<ov::op::v3::ScatterUpdate>(dst_flat, indices, src_flat, axis);

    ಸ್ವಯಂಚಾಲಿತ dst_shape = std::make_shared<ov::op::v3::Shape_ದ>(dst, ov::element::i64);
    ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v1::Reshape>(updated_flat, dst_shape, ಸುಳ್ಳು);

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
