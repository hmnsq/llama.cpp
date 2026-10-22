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

Output_ಸದಿಶ translate_pool_2d(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    ಸ್ಥಿರಾಂಕ int32_t * params = context.get_output_op_params();

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k0 = params[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k1 = params[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s0 = params[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s1 = params[4];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p0 = params[5];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p1 = params[6];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ op_case = context.get_op_case();
    ov::Output<Node> ಒಳಹರಿವು = context.get_ಒಳಹರಿವು(0);
    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::Shape pads_begin{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape pads_ಕೊನೆ{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape kernel{static_cast<size_t>(k1), static_cast<size_t>(k0)};
    ov::Output<Node> res;

    ಆಯ್ಕೆ (op_case) {
    ಸ್ಥಿತಿ 1:  // GGML_OP_POOL_MAX
    {
        res = std::make_shared<ov::op::v1::MaxPool>(ಒಳಹರಿವು, strides, pads_begin, pads_ಕೊನೆ, kernel);
        ನಿಲ್ಲಿಸು;
    }
    ಸ್ಥಿತಿ 2:  // GGML_OP_POOL_AVG
    {
        res = std::make_shared<ov::op::v1::AvgPool>(ಒಳಹರಿವು, strides, pads_begin, pads_ಕೊನೆ, kernel, ಸುಳ್ಳು);
        ನಿಲ್ಲಿಸು;
    }
    ಪೂರ್ವನಿಯೋಜಿತ:
        ನಿಲ್ಲಿಸು;
    }
    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
