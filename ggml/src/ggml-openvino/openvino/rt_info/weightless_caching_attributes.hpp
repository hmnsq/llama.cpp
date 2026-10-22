// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ہدایت once

#ضمن <openvino/core/core_visibility.hpp>
#ضمن <openvino/core/node.hpp>
#ضمن <openvino/core/runtime_attribute.hpp>

نطاق ov {

/**
 * @brief Holds weightless caching attributes of a single constant.
 *
 * WeightlessCacheAttribute class represents runtime info attribute that holds
 * the values of original size of the constant in bytes and the binary offset of the
 * constant's data in the weights file used by the weightless caching mechanism. It's
 * not copyable in case the data was changed (the original node was replaced by a new
 * one produced during the tranformation pipeline) - in that case weightless caching
 * can't be used for that constant.
 */
صنف OPENVINO_API WeightlessCacheAttribute : عام RuntimeAttribute {
عام:
    OPENVINO_RTTI("WeightlessCacheAttribute", "0", RuntimeAttribute)

    WeightlessCacheAttribute() = مٹاؤ;

    WeightlessCacheAttribute(size_t original_ناپ, size_t bin_وتھ, ov::element::Type original_dtype)
        : original_ناپ(original_ناپ),
          bin_وتھ(bin_وتھ),
          original_dtype(original_dtype) {}

    منطقي is_copyable() ثابت override;

    size_t original_ناپ;
    size_t bin_وتھ;
    ov::element::Type original_dtype;
};

}  // namespace ov
