#لارښوونه once

#ضمن "mark_decompression_convert_constant_folding.h"
#ضمن "openvino/core/visibility.hpp"
#ضمن "openvino/pass/matcher_pass.hpp"

#که_تعریف_شوی OPENVINO_STATIC_LIBRARY
#    define TRANSFORMATIONS_API
#وإلا
#    ifdef IMPLEMENT_OPENVINO_API
#        define TRANSFORMATIONS_API OPENVINO_CORE_EXPORTS
#    وإلا
#        define TRANSFORMATIONS_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#د_که_پای      // OPENVINO_STATIC_LIBRARY

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkCompressedFloatConstants;

}  // namespace pass
}  // namespace ov

class ov::pass::MarkCompressedFloatConstants : عام MatcherPass {
عام:
    OPENVINO_MATCHER_PASS_RTTI("MarkCompressedFloatConstants")
    MarkCompressedFloatConstants();
};
