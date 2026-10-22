#ہدایت once

#شامل "mark_decompression_convert_constant_folding.h"
#شامل "openvino/core/visibility.hpp"
#شامل "openvino/pass/matcher_pass.hpp"

#اگر_تعریف_شدہ OPENVINO_STATIC_LIBRARY
#    define TRANSFORMATIONS_API
#ورنہ
#    ifdef IMPLEMENT_OPENVINO_API
#        define TRANSFORMATIONS_API OPENVINO_CORE_EXPORTS
#    ورنہ
#        define TRANSFORMATIONS_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#اگر_ختم      // OPENVINO_STATIC_LIBRARY

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkCompressedFloatConstants;

}  // namespace pass
}  // namespace ov

class ov::pass::MarkCompressedFloatConstants : عوامی MatcherPass {
عوامی:
    OPENVINO_MATCHER_PASS_RTTI("MarkCompressedFloatConstants")
    MarkCompressedFloatConstants();
};
