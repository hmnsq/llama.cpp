#അന്തര്ഭാവ "ggml-vulkan-common.h"

#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <cstdlib>
#അന്തര്ഭാവ <mutex>
#അന്തര്ഭാവ <shared_mutex>

സ്ഥിതം ശൂന്യം * portable_aligned_malloc(size_t align, size_t size) {
#ifdef _WIN32
    തിരികെ _aligned_malloc(size, align);
#അല്ലെങ്കിൽ
    ശൂന്യം * ptr = nullptr;
    എങ്കിൽ (posix_memalign(&ptr, align, size) != 0) {
        തിരികെ nullptr;
    }
    തിരികെ ptr;
#endif
}

സ്ഥിതം ശൂന്യം portable_aligned_free(ശൂന്യം * ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#അല്ലെങ്കിൽ
    std::സ്വതന്ത്രം(ptr);
#endif
}

ggml_backend_buffer_type_i ggml_backend_vk_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_vk_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_vk_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ശൂന്യ,
    /* .get_alignment       = */ ggml_backend_vk_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_vk_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_vk_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ശൂന്യ,
    /* .is_host             = */ ശൂന്യ,
};

സ്ഥിതം std::vector<uint32_t> ggml_vk_find_memory_properties(സ്ഥിരം vk::PhysicalDeviceMemoryProperties* mem_props, vk::MemoryRequirements* mem_req, vk::MemoryPropertyFlags flags) {
    std::vector<uint32_t> indices;

    വേണ്ടി (uint32_t i = 0; i < mem_props->memory_Type_എണ്ണം; ++i) {
        vk::MemoryType memory_type = mem_props->memoryTypes[i];
        എങ്കിൽ ((mem_req->memoryTypeBits & ((uint64_t)1 << i)) &&
            (flags & memory_type.propertyFlags) == flags &&
            mem_props->memoryHeaps[memory_type.heapIndex].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    തിരികെ indices;
}

സ്ഥിതം vk_buffer ggml_vk_create_buffer(vk_device& device, size_t size, സ്ഥിരം std::initializer_list<vk::MemoryPropertyFlags> & req_flags_list,
                                       ശൂന്യം *import_ptr = nullptr) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    എങ്കിൽ (size > device->max_buffer_size) {
        എറിയുക vk::OutOfDeviceMemoryError("Requested buffer size exceeds device buffer size limit");
    }

    vk_buffer buf = std::make_shared<vk_buffer_struct>();

    എങ്കിൽ (size == 0) {
        buf->size = 0;
        തിരികെ buf;
    }

    vk::BufferUsageFlags usage_flags = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    vk::MemoryAllocateFlags mem_flags {};
    എങ്കിൽ (device->buffer_device_address) {
        usage_flags |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
        mem_flags |= vk::MemoryAllocateFlagBits::eDeviceAddress;
    }

    vk::BufferCreateInfo buffer_create_info{
        vk::BufferCreateFlags(),
        size,
        usage_flags,
        vk::SharingMode::eExclusive,
        0,
        nullptr,
    };

    vk::ExternalMemoryBufferCreateInfo external_memory_bci;
    എങ്കിൽ (import_ptr) {
        external_memory_bci.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
        buffer_create_info.setPNext(&external_memory_bci);
    }

    buf->buffer = device->device.createBuffer(buffer_create_info);

    vk::MemoryRequirements mem_req = device->device.getBufferMemoryRequirements(buf->buffer);

    vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

    സ്ഥിരം vk::MemoryPriorityAllocateInfoEXT mem_priority_info { 1.0f };

    vk::MemoryAllocateFlagsInfo mem_flags_info { mem_flags };

    എങ്കിൽ (device->memory_priority) {
        mem_flags_info.setPNext(&mem_priority_info);
    }

    എങ്കിൽ (import_ptr) {
        vk::MemoryHostPointerPropertiesEXT host_pointer_props;
        ശ്രമിക്കുക {
            host_pointer_props = device->device.getMemoryHostPointerPropertiesEXT(vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, import_ptr);
        } പിടിക്കുക (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroyBuffer(buf->buffer);
            തിരികെ {};
        }
        vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

        uint32_t memory_type_idx;
        vk::MemoryPropertyFlags property_flags = *req_flags_list.begin();
        വേണ്ടി (memory_type_idx = 0; memory_type_idx < 32; ++memory_type_idx) {
            എങ്കിൽ (!(host_pointer_props.memoryTypeBits & (1u << memory_type_idx))) {
                തുടരുക;
            }
            എങ്കിൽ (!(mem_req.memoryTypeBits & (1u << memory_type_idx))) {
                തുടരുക;
            }

            vk::MemoryType memory_type = mem_props.memoryTypes[memory_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            എങ്കിൽ ((memory_type.propertyFlags & property_flags) == property_flags) {
                property_flags = memory_type.propertyFlags;
                നിർത്തുക;
            }
        }
        എങ്കിൽ (memory_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroyBuffer(buf->buffer);
            തിരികെ {};
        }

        buf->memory_property_flags = mem_props.memoryTypes[memory_type_idx].propertyFlags;
        ശ്രമിക്കുക {
            vk::ImportMemoryHostPointerInfoEXT import_info;
            import_info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
            import_info.pHostPointer = import_ptr;
            import_info.setPNext(&mem_flags_info);
            buf->device_memory = device->device.allocateMemory({ size, memory_type_idx, &import_info });
        } പിടിക്കുക (സ്ഥിരം vk::SystemError& e) {
        }
    } അല്ലെങ്കിൽ {
        വേണ്ടി (auto it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            സ്ഥിരം auto & req_flags = *it;

            സ്ഥിരം std::vector<uint32_t> memory_type_indices = ggml_vk_find_memory_properties(&mem_props, &mem_req, req_flags);

            എങ്കിൽ (memory_type_indices.empty()) {
                തുടരുക;
            }

            തർക്കം done = അസത്യം;

            വേണ്ടി (auto mtype_it = memory_type_indices.begin(); mtype_it != memory_type_indices.end(); mtype_it++) {
                ശ്രമിക്കുക {
                    buf->device_memory = device->device.allocateMemory({ mem_req.size, *mtype_it, &mem_flags_info });
                    buf->memory_property_flags = mem_props.memoryTypes[*mtype_it].propertyFlags;
                    done = സത്യം;
                    നിർത്തുക;
                } പിടിക്കുക (സ്ഥിരം vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    എങ്കിൽ (it + 1 == req_flags_list.end() && mtype_it + 1 == memory_type_indices.end()) {
                        device->device.destroyBuffer(buf->buffer);
                        എറിയുക e;
                    }
                }
            }

            എങ്കിൽ (done) {
                നിർത്തുക;
            }
        }
    }

    എങ്കിൽ (!buf->device_memory) {
        device->device.destroyBuffer(buf->buffer);
        എറിയുക vk::OutOfDeviceMemoryError("No suitable memory type found");
    }

    buf->ptr = nullptr;

    എങ്കിൽ (import_ptr) {
        buf->ptr = import_ptr;
    } അല്ലെങ്കിൽ {
        എങ്കിൽ (buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
            buf->ptr = device->device.mapMemory(buf->device_memory, 0, VK_WHOLE_SIZE);
        }
    }

    device->device.bindBufferMemory(buf->buffer, buf->device_memory, 0);

    buf->device = device;
    buf->size = size;

    എങ്കിൽ (device->buffer_device_address) {
        സ്ഥിരം vk::BufferDeviceAddressInfo addressInfo(buf->buffer);
        buf->bda_addr = device->device.getBufferAddress(addressInfo);
    }

    device->memory_logger->log_allocation(buf, size);

    തിരികെ buf;
}

vk_buffer ggml_vk_create_buffer_check(vk_device& device, size_t size, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags) {
    ശ്രമിക്കുക {
        തിരികെ ggml_vk_create_buffer(device, size, {req_flags, fallback_flags});
    } പിടിക്കുക (സ്ഥിരം vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        എറിയുക e;
    }
}

vk_buffer ggml_vk_create_buffer_device(vk_device& device, size_t size) {
    vk_buffer buf;
    ശ്രമിക്കുക {
        എങ്കിൽ (device->prefer_host_memory) {
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal});
        } അല്ലെങ്കിൽ എങ്കിൽ (device->uma) {
            // On UMA, prefer host-visible memory so direct tensor borrowing works.
            // If unavailable, fall back to device-local memory.
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
        } അല്ലെങ്കിൽ എങ്കിൽ (device->disable_host_visible_vidmem) {
            എങ്കിൽ (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } അല്ലെങ്കിൽ {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        } അല്ലെങ്കിൽ {
            // use rebar if available, otherwise fallback to device only visible memory
            എങ്കിൽ (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } അല്ലെങ്കിൽ {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        }
    } പിടിക്കുക (സ്ഥിരം vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        എറിയുക e;
    }

    തിരികെ buf;
}

ശൂന്യം ggml_vk_destroy_buffer(vk_buffer& buf) {
    എങ്കിൽ (buf == nullptr) {
        തിരികെ;
    }

    എങ്കിൽ (buf->device != nullptr) {
        buf->device->memory_logger->log_deallocation(buf);
    }

    buf.reset();
}

// caller must hold host_allocations_mutex
സ്ഥിതം size_t ggml_vk_host_allocation_find(സ്ഥിരം ശൂന്യം * ptr) {
    വേണ്ടി (size_t i = 0; i < vk_instance.host_allocations.size(); i++) {
        സ്ഥിരം uint8_t * base = (സ്ഥിരം uint8_t *) vk_instance.host_allocations[i].base;
        എങ്കിൽ (ptr >= base && ptr < base + vk_instance.host_allocations[i].size) {
            തിരികെ i;
        }
    }
    തിരികെ SIZE_MAX;
}

ശൂന്യം * ggml_vk_host_malloc(size_t size) {
    VK_LOG_MEMORY("ggml_vk_host_malloc(" << size << ")");

    ggml_vk_instance_init();

    എങ്കിൽ (vk_instance.multi_device) {
        സ്ഥിരം size_t align      = std::max<size_t>(vk_instance.host_import_alignment, TENSOR_ALIGNMENT);
        സ്ഥിരം size_t alloc_size = (size + align - 1) / align * align;

        ശൂന്യം * ptr = portable_aligned_malloc(align, alloc_size);
        എങ്കിൽ (ptr == nullptr) {
            തിരികെ nullptr;
        }
        vk_host_allocation entry;
        entry.base = ptr;
        entry.size = alloc_size;

        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        vk_instance.host_allocations.push_back(std::move(entry));
        തിരികെ ptr;
    }

    vk_device dev0 = ggml_vk_get_device(0);
    vk_buffer buf;
    ശ്രമിക്കുക {
        buf = ggml_vk_create_buffer(dev0, size,
            {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
    } പിടിക്കുക (vk::SystemError&) {
        തിരികെ nullptr;
    }

    എങ്കിൽ (!(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n", size / 1024.0 / 1024.0);
        ggml_vk_destroy_buffer(buf);
        തിരികെ nullptr;
    }

    vk_host_allocation entry;
    entry.base             = buf->ptr;
    entry.size             = size;
    entry.views[dev0->idx] = buf;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
    vk_instance.host_allocations.push_back(std::move(entry));
    തിരികെ buf->ptr;
}

ശൂന്യം ggml_vk_host_free(ശൂന്യം * ptr) {
    എങ്കിൽ (ptr == nullptr) {
        തിരികെ;
    }
    VK_LOG_MEMORY("ggml_vk_host_free(" << ptr << ")");

    vk_host_allocation entry;
    {
        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        സ്ഥിരം size_t index = ggml_vk_host_allocation_find(ptr);
        എങ്കിൽ (index == SIZE_MAX) {
            fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
            തിരികെ;
        }
        entry = std::move(vk_instance.host_allocations[index]);
        vk_instance.host_allocations.erase(vk_instance.host_allocations.begin() + index);
    }

    വേണ്ടി (auto & view : entry.views) {
        ggml_vk_destroy_buffer(view);
    }
    എങ്കിൽ (vk_instance.multi_device) {
        portable_aligned_free(entry.base);
    }
}

ശൂന്യം ggml_vk_host_get(സ്ഥിരം vk_device& device, സ്ഥിരം ശൂന്യം * ptr, vk_buffer& buf, size_t& buf_offset) {
    buf = nullptr;
    buf_offset = 0;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);

    സ്ഥിരം size_t index = ggml_vk_host_allocation_find(ptr);
    എങ്കിൽ (index == SIZE_MAX) {
        തിരികെ;
    }
    vk_host_allocation & entry = vk_instance.host_allocations[index];

    എങ്കിൽ (entry.views[device->idx] == nullptr) {
        സ്ഥിരം uint32_t bit = 1u << device->idx;
        എങ്കിൽ (!vk_instance.multi_device || !vk_instance.device_host_buffer_import[device->idx] || (vk_instance.host_import_failed & bit)) {
            തിരികെ;
        }
        GGML_ASSERT(device->external_memory_host);
        // const only propagates from the const context; the pointee is mutable
        entry.views[device->idx] = ggml_vk_buffer_from_host_ptr(const_cast<vk_device&>(device), entry.base, entry.size);
        എങ്കിൽ (entry.views[device->idx] == nullptr) {
            vk_instance.host_import_failed |= bit;
            GGML_LOG_WARN("ggml_vulkan: %s: host memory import failed, using staging\n", device->name.c_str());
            തിരികെ;
        }
    }

    buf = entry.views[device->idx];
    buf_offset = (സ്ഥിരം uint8_t *) ptr - (സ്ഥിരം uint8_t *) entry.base;
}

ശൂന്യം ggml_vk_ensure_sync_staging_buffer(vk_device& device, size_t size) {
    എങ്കിൽ (device->sync_staging == nullptr || device->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(device->sync_staging);
        device->sync_staging = ggml_vk_create_buffer_check(device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

ശൂന്യം ggml_vk_ensure_sync_staging_buffer(ggml_backend_vk_context * ctx, size_t size) {
    എങ്കിൽ (ctx->sync_staging == nullptr || ctx->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_buffer_check(ctx->device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

സ്ഥിതം ശൂന്യം ggml_vk_buffer_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_buffer& dst, size_t offset, സ്ഥിരം ggml_tensor * tensor, തർക്കം sync_staging = അസത്യം) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    എങ്കിൽ(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::endl;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(ctx->device, tensor->data, buf, buf_offset);

    സ്ഥിരം uint64_t ne0 = tensor->ne[0];
    സ്ഥിരം uint64_t ne1 = tensor->ne[1];
    സ്ഥിരം uint64_t ne2 = tensor->ne[2];
    സ്ഥിരം uint64_t ne3 = tensor->ne[3];
    സ്ഥിരം uint64_t nb0 = tensor->nb[0];
    സ്ഥിരം uint64_t nb1 = tensor->nb[1];
    സ്ഥിരം uint64_t nb2 = tensor->nb[2];
    സ്ഥിരം uint64_t nb3 = tensor->nb[3];
    സ്ഥിരം ggml_type type = tensor->type;
    സ്ഥിരം uint64_t ts = ggml_type_size(type);
    സ്ഥിരം uint64_t bs = ggml_blck_size(type);

    സ്ഥിരം uint64_t dstnb0 = ts;
    സ്ഥിരം uint64_t dstnb1 = dstnb0*(ne0/bs);
    സ്ഥിരം uint64_t dstnb2 = dstnb1*ne1;
    സ്ഥിരം uint64_t dstnb3 = dstnb2*ne2;

    സ്ഥിരം uint64_t ne = ggml_nelements(tensor);

    എങ്കിൽ (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices;

        വേണ്ടി (uint64_t i3 = 0; i3 < ne3; i3++) {
            വേണ്ടി (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                എങ്കിൽ (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_offset + i3*nb3 + i2*nb2, offset + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } അല്ലെങ്കിൽ {
                    വേണ്ടി (uint64_t i1 = 0; i1 < ne1; i1++) {
                        എങ്കിൽ (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_offset + i3*nb3 + i2*nb2 + i1*nb1, offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } അല്ലെങ്കിൽ {
                            സ്ഥിരം uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                            സ്ഥിരം uint64_t d_off = offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            വേണ്ടി (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i0*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(ctx, subctx);
        subctx->s->buffer->buf.copyBuffer(buf->buffer, dst->buffer, slices);
        തിരികെ;
    }

    എങ്കിൽ (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_buffer& staging = ctx->device->sync_staging;
    സ്ഥിരം uint64_t copy_size = ts*ne/bs;
    ggml_vk_ensure_sync_staging_buffer(ctx->device, copy_size);
    VkBufferCopy buf_copy{ 0, offset, copy_size };

    ggml_vk_sync_buffers(ctx, subctx);
    vkCmdCopyBuffer(subctx->s->buffer->buf, (VkBuffer)staging->buffer, (VkBuffer)dst->buffer, 1, &buf_copy);

    വേണ്ടി (uint64_t i3 = 0; i3 < ne3; i3++) {
        വേണ്ടി (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            എങ്കിൽ (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2, (സ്ഥിരം uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2, dstnb2, &subctx->in_memcpys);
            } അല്ലെങ്കിൽ {
                വേണ്ടി (uint64_t i1 = 0; i1 < ne1; i1++) {
                    എങ്കിൽ (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (സ്ഥിരം uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->in_memcpys);
                    } അല്ലെങ്കിൽ {
                        സ്ഥിരം uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                        സ്ഥിരം uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        വേണ്ടി (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (സ്ഥിരം uint8_t *) tensor->data + s_off + i0*nb0, dstnb0, &subctx->in_memcpys);
                        }
                    }
                }
            }
        }
    }
}

തർക്കം ggml_vk_buffer_write_2d_async(vk_context subctx, vk_buffer& dst, size_t offset, സ്ഥിരം ശൂന്യം * src, size_t spitch, size_t dpitch, size_t width, size_t height, തർക്കം sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << width << ", " << height << ")");
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_offset);

    എങ്കിൽ (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices(1);
        എങ്കിൽ (width == spitch && width == dpitch) {
            // Only do single write if stride is equal
            slices[0].srcOffset = buf_offset;
            slices[0].dstOffset = offset;
            slices[0].size = width * height;
        } അല്ലെങ്കിൽ {
            slices.resize(height);
            വേണ്ടി (size_t i = 0; i < height; i++) {
                slices[i].srcOffset = buf_offset + i * spitch;
                slices[i].dstOffset = offset + i * dpitch;
                slices[i].size = width;
            }
        }

        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->buffer->buf.copyBuffer(buf->buffer, dst->buffer, slices);
        തിരികെ സത്യം;
    }
    VK_LOG_DEBUG("STAGING");

    എങ്കിൽ (!sync_staging) {
        // copy was not handled caller needs to fall back
        തിരികെ അസത്യം;
    }

    // Staging buffer required
    സ്ഥിരം size_t staging_size = width * height;
    ggml_vk_ensure_sync_staging_buffer(dst->device, staging_size);

    vk_buffer& staging_buffer = dst->device->sync_staging;

    std::vector<vk::BufferCopy> slices(1);
    എങ്കിൽ (width == dpitch) {
        slices[0].srcOffset = 0;
        slices[0].dstOffset = offset;
        slices[0].size = staging_size;
    } അല്ലെങ്കിൽ {
        slices.resize(height);
        വേണ്ടി (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = i * width;
            slices[i].dstOffset = offset + i * dpitch;
            slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer->buf.copyBuffer(staging_buffer->buffer, dst->buffer, slices);

    എങ്കിൽ (width == spitch) {
        deferred_memcpy((uint8_t *)staging_buffer->ptr, src, staging_size, &subctx->in_memcpys);
    } അല്ലെങ്കിൽ {
        വേണ്ടി (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *)staging_buffer->ptr + i * width, (സ്ഥിരം uint8_t *) src + i * spitch, width, &subctx->in_memcpys);
        }
    }
    തിരികെ സത്യം;
}

തർക്കം ggml_vk_buffer_write_async(vk_context subctx, vk_buffer& dst, size_t offset, സ്ഥിരം ശൂന്യം * src, size_t size, തർക്കം sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    തിരികെ ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, size, size, size, 1, sync_staging);
}

ശൂന്യം ggml_vk_buffer_write_2d(vk_buffer& dst, size_t offset, സ്ഥിരം ശൂന്യം * src, size_t spitch, size_t dpitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << width << ", " << height << ")");
    // Buffer is already mapped
    എങ്കിൽ(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        GGML_ASSERT(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        എങ്കിൽ (width == spitch && width == dpitch) {
            memcpy((uint8_t *)dst->ptr + offset, src, width * height);
        } അല്ലെങ്കിൽ {
            വേണ്ടി (size_t i = 0; i < height; i++) {
                memcpy((uint8_t *)dst->ptr + offset + i * dpitch, (സ്ഥിരം uint8_t *) src + i * spitch, width);
            }
        }
    } അല്ലെങ്കിൽ {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(dst->device, subctx);
        തർക്കം ret = ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, spitch, dpitch, width, height, സത്യം);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        വേണ്ടി (auto& cpy : subctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        വേണ്ടി (auto& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, സത്യം, UINT64_MAX), "vk_buffer_write_2d waitForFences", dst->device);
        dst->device->device.resetFences({ dst->device->fence });
        ggml_vk_queue_command_pools_cleanup(dst->device);
    }
}

ശൂന്യം ggml_vk_buffer_write(vk_buffer& dst, size_t offset, സ്ഥിരം ശൂന്യം * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_buffer_write_2d(dst, offset, src, size, size, size, 1);
}

തർക്കം ggml_vk_buffer_read_2d_async(vk_context subctx, vk_buffer& src, size_t offset, ശൂന്യം * dst, size_t spitch, size_t dpitch, size_t width, size_t height, തർക്കം sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << offset << ", width=" << width << ", height=" << height << ")");
    GGML_ASSERT(width > 0);
    GGML_ASSERT(height > 0);
    GGML_ASSERT(src != nullptr);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_offset);

    std::vector<vk::BufferCopy> slices(1);
    എങ്കിൽ (width == spitch && width == dpitch) {
        // Only do single write if stride is equal
        slices[0].srcOffset = offset;
        slices[0].dstOffset = buf_offset;
        slices[0].size = width * height;
    } അല്ലെങ്കിൽ {
        slices.resize(height);
        വേണ്ടി (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = offset + i * spitch;
            slices[i].dstOffset = buf_offset + i * dpitch;
            slices[i].size = width;
        }
    }

    എങ്കിൽ (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->buffer->buf.copyBuffer(src->buffer, buf->buffer, slices);

        തിരികെ സത്യം;
    }
    VK_LOG_DEBUG("STAGING");

    എങ്കിൽ (!sync_staging) {
        // copy was not handled caller needs to fall back
        തിരികെ അസത്യം;
    }

    // Fall back to staging buffer
    സ്ഥിരം size_t staging_size = width * height;
    ggml_vk_ensure_sync_staging_buffer(src->device, staging_size);

    vk_buffer& staging_buffer = src->device->sync_staging;

    std::vector<vk::BufferCopy> staging_slices(1);
    എങ്കിൽ (width == spitch) {
        staging_slices[0].srcOffset = offset;
        staging_slices[0].dstOffset = 0;
        staging_slices[0].size = staging_size;
    } അല്ലെങ്കിൽ {
        staging_slices.resize(height);
        വേണ്ടി (size_t i = 0; i < height; i++) {
            staging_slices[i].srcOffset = offset + i * spitch;
            staging_slices[i].dstOffset = i * width;
            staging_slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer->buf.copyBuffer(src->buffer, staging_buffer->buffer, staging_slices);

    എങ്കിൽ (width == dpitch) {
        deferred_memcpy(dst, staging_buffer->ptr, staging_size, &subctx->out_memcpys);
    } അല്ലെങ്കിൽ {
        വേണ്ടി (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (സ്ഥിരം uint8_t *) staging_buffer->ptr + i * width, width, &subctx->out_memcpys);
        }
    }
    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം ggml_vk_buffer_read_async(vk_context subctx, vk_buffer& src, size_t offset, ശൂന്യം * dst, size_t size, തർക്കം sync_staging = അസത്യം) {
    തിരികെ ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, size, size, size, 1, sync_staging);
}

ശൂന്യം ggml_vk_buffer_read_2d(vk_buffer& src, size_t offset, ശൂന്യം * dst, size_t spitch, size_t dpitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->buffer << ", " << offset << ", " << width << ", " << height << ")");

    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    // AMD UMA: uncached host-visible memory is write-combined, CPU reads are slow
    സ്ഥിരം തർക്കം slow_host_read = src->device->vendor_id == VK_VENDOR_ID_AMD &&
                                !(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCached);
    എങ്കിൽ(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible && src->device->uma && !slow_host_read) {
        GGML_ASSERT(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        vk_context subctx = ggml_vk_create_temporary_context(src->device->compute_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        subctx->s->buffer->buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eHost,
            {},
            { { vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eHostRead } },
            {}, {});
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, സത്യം, UINT64_MAX),
                 "vk_buffer_read_2d uma waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        എങ്കിൽ (width == spitch && width == dpitch) {
            memcpy(dst, (സ്ഥിരം uint8_t *) src->ptr + offset, width * height);
        } അല്ലെങ്കിൽ {
            വേണ്ടി (size_t i = 0; i < height; i++) {
                memcpy((uint8_t *) dst + i * dpitch, (സ്ഥിരം uint8_t *) src->ptr + offset + i * spitch, width);
            }
        }
    } അല്ലെങ്കിൽ {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        തർക്കം ret = ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, spitch, dpitch, width, height, സത്യം);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, സത്യം, UINT64_MAX), "vk_buffer_read_2d waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        വേണ്ടി (auto& cpy : subctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
    }
}

ശൂന്യം ggml_vk_buffer_read(vk_buffer& src, size_t offset, ശൂന്യം * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->buffer << ", " << offset << ", " << size << ")");
    ggml_vk_buffer_read_2d(src, offset, dst, size, size, size, 1);
}

ശൂന്യം ggml_vk_buffer_copy_async(vk_context& ctx, vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    VkBufferCopy bc{ src_offset, dst_offset, size };

    vkCmdCopyBuffer(ctx->s->buffer->buf, (VkBuffer)src->buffer, (VkBuffer)dst->buffer, 1, &bc);
}

ശൂന്യം ggml_vk_buffer_copy(vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    എങ്കിൽ (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_buffer_copy_async(subctx, dst, dst_offset, src, src_offset, size);
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, സത്യം, UINT64_MAX), "vk_buffer_copy waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);
    } അല്ലെങ്കിൽ {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_buffer(src->device, size);

        // Copy to src staging buffer
        ggml_vk_buffer_copy(src->device->sync_staging, 0, src, src_offset, size);
        // Copy to dst buffer
        ggml_vk_buffer_write(dst, dst_offset, src->device->sync_staging->ptr, size);
    }
}

ശൂന്യം ggml_vk_buffer_memset_async(vk_context& ctx, vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << offset << ", " << c << ", " << size << ")");

    എങ്കിൽ (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->ptr + offset, c, size, &ctx->memsets);
        തിരികെ;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->buffer->buf.fillBuffer(dst->buffer, offset, size, c);
}

ശൂന്യം ggml_vk_buffer_memset(vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << offset << ", " << c << ", " << size << ")");

    എങ്കിൽ (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        memset((uint8_t*)dst->ptr + offset, c, size);
        തിരികെ;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue->cmd_pool);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->buffer->buf.fillBuffer(dst->buffer, offset, size, c);
    ggml_vk_ctx_end(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, സത്യം, UINT64_MAX), "vk_memset waitForFences", dst->device);
    dst->device->device.resetFences({ dst->device->fence });
    ggml_vk_queue_command_pools_cleanup(dst->device);
}

ggml_backend_buffer_i ggml_backend_vk_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_vk_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_vk_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_vk_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_vk_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_buffer_clear,
    /* .reset           = */ ശൂന്യ,
};

vk_buffer ggml_vk_buffer_from_host_ptr(vk_device & device, ശൂന്യം * ptr, size_t size) {
    എങ്കിൽ (!device->external_memory_host) {
        തിരികെ {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    എങ്കിൽ (uptr & (device->min_imported_host_pointer_alignment - 1)) {
        തിരികെ {};
    }
    എങ്കിൽ (size & (device->min_imported_host_pointer_alignment - 1)) {
        തിരികെ {};
    }

    സ്ഥിരം vk::MemoryPropertyFlags property_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached;

    vk_buffer buf {};
    ശ്രമിക്കുക {
        buf = ggml_vk_create_buffer(device, size, { property_flags }, ptr);
    } പിടിക്കുക (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed ggml_vk_create_buffer (%s)\n", e.what());
    }

    തിരികെ buf;
}

