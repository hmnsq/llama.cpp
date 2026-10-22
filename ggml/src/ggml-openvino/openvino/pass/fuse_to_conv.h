#అన్తర్భావ "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class Fuse_గమ్యం_Conv : బహిరంగ ov::pass::MatcherPass {
బహిరంగ:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseToConv")
    Fuse_గమ్యం_Conv();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
