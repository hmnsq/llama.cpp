#شامل "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class Fuse_مقصد_SDPA : عوامی ov::pass::MatcherPass {
عوامی:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseToSDPA")
    Fuse_مقصد_SDPA();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
