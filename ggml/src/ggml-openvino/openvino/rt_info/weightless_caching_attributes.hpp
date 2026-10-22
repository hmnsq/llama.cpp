// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#సూచన once

#అన్తర్భావ <openvino/core/core_visibility.hpp>
#అన్తర్భావ <openvino/core/node.hpp>
#అన్తర్భావ <openvino/core/runtime_attribute.hpp>

నామాకాశ ov {

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
తరగతి OPENVINO_API WeightlessCacheAttribute : బహిరంగ RuntimeAttribute {
బహిరంగ:
    OPENVINO_RTTI("WeightlessCacheAttribute", "0", RuntimeAttribute)

    WeightlessCacheAttribute() = తొలగించు;

    WeightlessCacheAttribute(size_t original_కొలత, size_t bin_స్థానభ్రంశం, ov::element::Type original_dtype)
        : original_కొలత(original_కొలత),
          bin_స్థానభ్రంశం(bin_స్థానభ్రంశం),
          original_dtype(original_dtype) {}

    తార్కిక is_copyable() స్థిరరాశి override;

    size_t original_కొలత;
    size_t bin_స్థానభ్రంశం;
    ov::element::Type original_dtype;
};

}  // namespace ov
