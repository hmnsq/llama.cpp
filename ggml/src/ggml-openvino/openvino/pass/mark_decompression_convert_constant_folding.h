#సూచన once

#అన్తర్భావ "mark_decompression_convert_constant_folding.h"
#అన్తర్భావ "openvino/core/visibility.hpp"
#అన్తర్భావ "openvino/pass/matcher_pass.hpp"

#నిర్వచితమైతే OPENVINO_STATIC_LIBRARY
#    define TRANSFORMATIONS_API
#లేకపోతే
#    ifdef IMPLEMENT_OPENVINO_API
#        define TRANSFORMATIONS_API OPENVINO_CORE_EXPORTS
#    లేకపోతే
#        define TRANSFORMATIONS_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#అయితే_ముగింపు      // OPENVINO_STATIC_LIBRARY

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkCompressedFloatConstants;

}  // namespace pass
}  // namespace ov

class ov::pass::MarkCompressedFloatConstants : బహిరంగ MatcherPass {
బహిరంగ:
    OPENVINO_MATCHER_PASS_RTTI("MarkCompressedFloatConstants")
    MarkCompressedFloatConstants();
};
