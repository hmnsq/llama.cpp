#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <openvino/op/avg_pool.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/max_pool.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

OutputVector translate_pool_2d(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    ಧ್ರುವ int32_t * params = context.get_output_op_params();

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k0 = params[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k1 = params[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s0 = params[3];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s1 = params[4];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ p0 = params[5];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ p1 = params[6];

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ op_case = context.get_op_case();
    ov::Output<Node> ಒಳಹರಿವು = context.get_ಒಳಹರಿವು(0);
    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::Shape pads_begin{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape pads_end{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape kernel{static_cast<size_t>(k1), static_cast<size_t>(k0)};
    ov::Output<Node> res;

    ವಿಕಲ್ಪ (op_case) {
    ಪಕ್ಷ 1:  // GGML_OP_POOL_MAX
    {
        res = std::make_shared<ov::op::v1::MaxPool>(ಒಳಹರಿವು, strides, pads_begin, pads_end, kernel);
        ಭಙ್ಗ;
    }
    ಪಕ್ಷ 2:  // GGML_OP_POOL_AVG
    {
        res = std::make_shared<ov::op::v1::AvgPool>(ಒಳಹರಿವು, strides, pads_begin, pads_end, kernel, ಅಸತ್ಯ);
        ಭಙ್ಗ;
    }
    ಸ್ವಭಾವ:
        ಭಙ್ಗ;
    }
    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
