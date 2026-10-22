// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ଅନ୍ତର୍ଭାଵ <openvino/frontend/frontend.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd {
ସର୍ବସାଧାରଣ:
    FrontEnd();

    ସ୍ଥିର std::shared_ptr<Model> convert(ଅଚଳ ନିବେଶ_Model::Ptr & model, bool naive = false);
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
