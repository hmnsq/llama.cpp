#شامل "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class SqueezeMatmul : عوامی ov::pass::MatcherPass {
عوامی:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::SqueezeMatmul")
    SqueezeMatmul();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
