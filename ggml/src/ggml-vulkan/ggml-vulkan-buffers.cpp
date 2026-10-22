#अन्तर्भाव "ggml-vulkan-common.h"

#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <mutex>
#अन्तर्भाव <shared_mutex>

स्थिर शून्य * portable_aligned_malloc(माप_प्रकार align, माप_प्रकार size) {
#ifdef _WIN32
    फर्काऊ _aligned_malloc(size, align);
#अन्यथा
    शून्य * ptr = nullptr;
    यदि (posix_memalign(&ptr, align, size) != 0) {
        फर्काऊ nullptr;
    }
    फर्काऊ ptr;
#endif
}

स्थिर शून्य portable_aligned_free(शून्य * ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#अन्यथा
    std::मुक्त(ptr);
#endif
}

ggml_backend_buffer_type_i ggml_backend_vk_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_vk_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_vk_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ रिक्त,
    /* .get_alignment       = */ ggml_backend_vk_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_vk_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_vk_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ रिक्त,
    /* .is_host             = */ रिक्त,
};

स्थिर std::vector<uint32_t> ggml_vk_find_memory_properties(अचर vk::PhysicalDeviceMemoryProperties* mem_props, vk::MemoryRequirements* mem_req, vk::MemoryPropertyFlags flags) {
    std::vector<uint32_t> indices;

    लागि (uint32_t i = 0; i < mem_props->memory_Type_गन्ती; ++i) {
        vk::MemoryType memory_type = mem_props->memoryTypes[i];
        यदि ((mem_req->memoryTypeBits & ((uint64_t)1 << i)) &&
            (flags & memory_type.propertyFlags) == flags &&
            mem_props->memoryHeaps[memory_type.heapIndex].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    फर्काऊ indices;
}

स्थिर vk_buffer ggml_vk_create_buffer(vk_device& device, माप_प्रकार size, अचर std::initializer_list<vk::MemoryPropertyFlags> & req_flags_list,
                                       शून्य *import_ptr = nullptr) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    यदि (size > device->max_buffer_size) {
        क्षिप vk::OutOfDeviceMemoryError("Requested buffer size exceeds device buffer size limit");
    }

    vk_buffer buf = std::make_shared<vk_buffer_struct>();

    यदि (size == 0) {
        buf->size = 0;
        फर्काऊ buf;
    }

    vk::BufferUsageFlags usage_flags = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    vk::MemoryAllocateFlags mem_flags {};
    यदि (device->buffer_device_address) {
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
    यदि (import_ptr) {
        external_memory_bci.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
        buffer_create_info.setPNext(&external_memory_bci);
    }

    buf->buffer = device->device.createBuffer(buffer_create_info);

    vk::MemoryRequirements mem_req = device->device.getBufferMemoryRequirements(buf->buffer);

    vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

    अचर vk::MemoryPriorityAllocateInfoEXT mem_priority_info { 1.0f };

    vk::MemoryAllocateFlagsInfo mem_flags_info { mem_flags };

    यदि (device->memory_priority) {
        mem_flags_info.setPNext(&mem_priority_info);
    }

    यदि (import_ptr) {
        vk::MemoryHostPointerPropertiesEXT host_pointer_props;
        यत्न {
            host_pointer_props = device->device.getMemoryHostPointerPropertiesEXT(vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, import_ptr);
        } गृहाण (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroyBuffer(buf->buffer);
            फर्काऊ {};
        }
        vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

        uint32_t memory_type_idx;
        vk::MemoryPropertyFlags property_flags = *req_flags_list.begin();
        लागि (memory_type_idx = 0; memory_type_idx < 32; ++memory_type_idx) {
            यदि (!(host_pointer_props.memoryTypeBits & (1u << memory_type_idx))) {
                जारी;
            }
            यदि (!(mem_req.memoryTypeBits & (1u << memory_type_idx))) {
                जारी;
            }

            vk::MemoryType memory_type = mem_props.memoryTypes[memory_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            यदि ((memory_type.propertyFlags & property_flags) == property_flags) {
                property_flags = memory_type.propertyFlags;
                रोक;
            }
        }
        यदि (memory_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroyBuffer(buf->buffer);
            फर्काऊ {};
        }

        buf->memory_property_flags = mem_props.memoryTypes[memory_type_idx].propertyFlags;
        यत्न {
            vk::ImportMemoryHostPointerInfoEXT import_info;
            import_info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
            import_info.pHostPointer = import_ptr;
            import_info.setPNext(&mem_flags_info);
            buf->device_memory = device->device.allocateMemory({ size, memory_type_idx, &import_info });
        } गृहाण (अचर vk::SystemError& e) {
        }
    } अन्यथा {
        लागि (स्वतः it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            अचर स्वतः & req_flags = *it;

            अचर std::vector<uint32_t> memory_type_indices = ggml_vk_find_memory_properties(&mem_props, &mem_req, req_flags);

            यदि (memory_type_indices.empty()) {
                जारी;
            }

            तार्किक done = असत्य;

            लागि (स्वतः mtype_it = memory_type_indices.begin(); mtype_it != memory_type_indices.end(); mtype_it++) {
                यत्न {
                    buf->device_memory = device->device.allocateMemory({ mem_req.size, *mtype_it, &mem_flags_info });
                    buf->memory_property_flags = mem_props.memoryTypes[*mtype_it].propertyFlags;
                    done = सत्य;
                    रोक;
                } गृहाण (अचर vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    यदि (it + 1 == req_flags_list.end() && mtype_it + 1 == memory_type_indices.end()) {
                        device->device.destroyBuffer(buf->buffer);
                        क्षिप e;
                    }
                }
            }

            यदि (done) {
                रोक;
            }
        }
    }

    यदि (!buf->device_memory) {
        device->device.destroyBuffer(buf->buffer);
        क्षिप vk::OutOfDeviceMemoryError("No suitable memory type found");
    }

    buf->ptr = nullptr;

    यदि (import_ptr) {
        buf->ptr = import_ptr;
    } अन्यथा {
        यदि (buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
            buf->ptr = device->device.mapMemory(buf->device_memory, 0, VK_WHOLE_SIZE);
        }
    }

    device->device.bindBufferMemory(buf->buffer, buf->device_memory, 0);

    buf->device = device;
    buf->size = size;

    यदि (device->buffer_device_address) {
        अचर vk::BufferDeviceAddressInfo addressInfo(buf->buffer);
        buf->bda_addr = device->device.getBufferAddress(addressInfo);
    }

    device->memory_logger->log_allocation(buf, size);

    फर्काऊ buf;
}

vk_buffer ggml_vk_create_buffer_check(vk_device& device, माप_प्रकार size, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags) {
    यत्न {
        फर्काऊ ggml_vk_create_buffer(device, size, {req_flags, fallback_flags});
    } गृहाण (अचर vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        क्षिप e;
    }
}

vk_buffer ggml_vk_create_buffer_device(vk_device& device, माप_प्रकार size) {
    vk_buffer buf;
    यत्न {
        यदि (device->prefer_host_memory) {
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal});
        } अन्यथा यदि (device->uma) {
            // On UMA, prefer host-visible memory so direct tensor borrowing works.
            // If unavailable, fall back to device-local memory.
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
        } अन्यथा यदि (device->disable_host_visible_vidmem) {
            यदि (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } अन्यथा {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        } अन्यथा {
            // use rebar if available, otherwise fallback to device only visible memory
            यदि (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } अन्यथा {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        }
    } गृहाण (अचर vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        क्षिप e;
    }

    फर्काऊ buf;
}

शून्य ggml_vk_destroy_buffer(vk_buffer& buf) {
    यदि (buf == nullptr) {
        फर्काऊ;
    }

    यदि (buf->device != nullptr) {
        buf->device->memory_logger->log_deallocation(buf);
    }

    buf.reset();
}

// caller must hold host_allocations_mutex
स्थिर माप_प्रकार ggml_vk_host_allocation_find(अचर शून्य * ptr) {
    लागि (माप_प्रकार i = 0; i < vk_instance.host_allocations.size(); i++) {
        अचर uint8_t * base = (अचर uint8_t *) vk_instance.host_allocations[i].base;
        यदि (ptr >= base && ptr < base + vk_instance.host_allocations[i].size) {
            फर्काऊ i;
        }
    }
    फर्काऊ SIZE_MAX;
}

शून्य * ggml_vk_host_malloc(माप_प्रकार size) {
    VK_LOG_MEMORY("ggml_vk_host_malloc(" << size << ")");

    ggml_vk_instance_init();

    यदि (vk_instance.multi_device) {
        अचर माप_प्रकार align      = std::max<माप_प्रकार>(vk_instance.host_import_alignment, TENSOR_ALIGNMENT);
        अचर माप_प्रकार alloc_size = (size + align - 1) / align * align;

        शून्य * ptr = portable_aligned_malloc(align, alloc_size);
        यदि (ptr == nullptr) {
            फर्काऊ nullptr;
        }
        vk_host_allocation entry;
        entry.base = ptr;
        entry.size = alloc_size;

        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        vk_instance.host_allocations.push_back(std::move(entry));
        फर्काऊ ptr;
    }

    vk_device dev0 = ggml_vk_get_device(0);
    vk_buffer buf;
    यत्न {
        buf = ggml_vk_create_buffer(dev0, size,
            {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
    } गृहाण (vk::SystemError&) {
        फर्काऊ nullptr;
    }

    यदि (!(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
        ख_लिखो(मानक_त्रुटि, "WARNING: failed to allocate %.2f MB of pinned memory\n", size / 1024.0 / 1024.0);
        ggml_vk_destroy_buffer(buf);
        फर्काऊ nullptr;
    }

    vk_host_allocation entry;
    entry.base             = buf->ptr;
    entry.size             = size;
    entry.views[dev0->idx] = buf;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
    vk_instance.host_allocations.push_back(std::move(entry));
    फर्काऊ buf->ptr;
}

शून्य ggml_vk_host_free(शून्य * ptr) {
    यदि (ptr == nullptr) {
        फर्काऊ;
    }
    VK_LOG_MEMORY("ggml_vk_host_free(" << ptr << ")");

    vk_host_allocation entry;
    {
        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        अचर माप_प्रकार index = ggml_vk_host_allocation_find(ptr);
        यदि (index == SIZE_MAX) {
            ख_लिखो(मानक_त्रुटि, "WARNING: failed to free pinned memory: memory not in map\n");
            फर्काऊ;
        }
        entry = std::move(vk_instance.host_allocations[index]);
        vk_instance.host_allocations.erase(vk_instance.host_allocations.begin() + index);
    }

    लागि (स्वतः & view : entry.views) {
        ggml_vk_destroy_buffer(view);
    }
    यदि (vk_instance.multi_device) {
        portable_aligned_free(entry.base);
    }
}

शून्य ggml_vk_host_get(अचर vk_device& device, अचर शून्य * ptr, vk_buffer& buf, माप_प्रकार& buf_offset) {
    buf = nullptr;
    buf_offset = 0;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);

    अचर माप_प्रकार index = ggml_vk_host_allocation_find(ptr);
    यदि (index == SIZE_MAX) {
        फर्काऊ;
    }
    vk_host_allocation & entry = vk_instance.host_allocations[index];

    यदि (entry.views[device->idx] == nullptr) {
        अचर uint32_t bit = 1u << device->idx;
        यदि (!vk_instance.multi_device || !vk_instance.device_host_buffer_import[device->idx] || (vk_instance.host_import_failed & bit)) {
            फर्काऊ;
        }
        GGML_ASSERT(device->external_memory_host);
        // const only propagates from the const context; the pointee is mutable
        entry.views[device->idx] = ggml_vk_buffer_from_host_ptr(const_cast<vk_device&>(device), entry.base, entry.size);
        यदि (entry.views[device->idx] == nullptr) {
            vk_instance.host_import_failed |= bit;
            GGML_LOG_WARN("ggml_vulkan: %s: host memory import failed, using staging\n", device->name.c_str());
            फर्काऊ;
        }
    }

    buf = entry.views[device->idx];
    buf_offset = (अचर uint8_t *) ptr - (अचर uint8_t *) entry.base;
}

शून्य ggml_vk_ensure_sync_staging_buffer(vk_device& device, माप_प्रकार size) {
    यदि (device->sync_staging == nullptr || device->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(device->sync_staging);
        device->sync_staging = ggml_vk_create_buffer_check(device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

शून्य ggml_vk_ensure_sync_staging_buffer(ggml_backend_vk_context * ctx, माप_प्रकार size) {
    यदि (ctx->sync_staging == nullptr || ctx->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_buffer_check(ctx->device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

स्थिर शून्य ggml_vk_buffer_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_buffer& dst, माप_प्रकार offset, अचर ggml_tensor * tensor, तार्किक sync_staging = असत्य) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    यदि(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::endl;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    माप_प्रकार buf_offset = 0;
    ggml_vk_host_get(ctx->device, tensor->data, buf, buf_offset);

    अचर uint64_t ne0 = tensor->ne[0];
    अचर uint64_t ne1 = tensor->ne[1];
    अचर uint64_t ne2 = tensor->ne[2];
    अचर uint64_t ne3 = tensor->ne[3];
    अचर uint64_t nb0 = tensor->nb[0];
    अचर uint64_t nb1 = tensor->nb[1];
    अचर uint64_t nb2 = tensor->nb[2];
    अचर uint64_t nb3 = tensor->nb[3];
    अचर ggml_type type = tensor->type;
    अचर uint64_t ts = ggml_type_size(type);
    अचर uint64_t bs = ggml_blck_size(type);

    अचर uint64_t dstnb0 = ts;
    अचर uint64_t dstnb1 = dstnb0*(ne0/bs);
    अचर uint64_t dstnb2 = dstnb1*ne1;
    अचर uint64_t dstnb3 = dstnb2*ne2;

    अचर uint64_t ne = ggml_nelements(tensor);

    यदि (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices;

        लागि (uint64_t i3 = 0; i3 < ne3; i3++) {
            लागि (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                यदि (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_offset + i3*nb3 + i2*nb2, offset + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } अन्यथा {
                    लागि (uint64_t i1 = 0; i1 < ne1; i1++) {
                        यदि (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_offset + i3*nb3 + i2*nb2 + i1*nb1, offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } अन्यथा {
                            अचर uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                            अचर uint64_t d_off = offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            लागि (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i0*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(ctx, subctx);
        subctx->s->buffer->buf.copyBuffer(buf->buffer, dst->buffer, slices);
        फर्काऊ;
    }

    यदि (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_buffer& staging = ctx->device->sync_staging;
    अचर uint64_t copy_size = ts*ne/bs;
    ggml_vk_ensure_sync_staging_buffer(ctx->device, copy_size);
    VkBufferCopy buf_copy{ 0, offset, copy_size };

    ggml_vk_sync_buffers(ctx, subctx);
    vkCmdCopyBuffer(subctx->s->buffer->buf, (VkBuffer)staging->buffer, (VkBuffer)dst->buffer, 1, &buf_copy);

    लागि (uint64_t i3 = 0; i3 < ne3; i3++) {
        लागि (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            यदि (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2, (अचर uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2, dstnb2, &subctx->in_memcpys);
            } अन्यथा {
                लागि (uint64_t i1 = 0; i1 < ne1; i1++) {
                    यदि (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (अचर uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->in_memcpys);
                    } अन्यथा {
                        अचर uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                        अचर uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        लागि (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (अचर uint8_t *) tensor->data + s_off + i0*nb0, dstnb0, &subctx->in_memcpys);
                        }
                    }
                }
            }
        }
    }
}

तार्किक ggml_vk_buffer_write_2d_async(vk_context subctx, vk_buffer& dst, माप_प्रकार offset, अचर शून्य * src, माप_प्रकार spitch, माप_प्रकार dpitch, माप_प्रकार width, माप_प्रकार height, तार्किक sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << width << ", " << height << ")");
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    माप_प्रकार buf_offset = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_offset);

    यदि (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices(1);
        यदि (width == spitch && width == dpitch) {
            // Only do single write if stride is equal
            slices[0].srcOffset = buf_offset;
            slices[0].dstOffset = offset;
            slices[0].size = width * height;
        } अन्यथा {
            slices.resize(height);
            लागि (माप_प्रकार i = 0; i < height; i++) {
                slices[i].srcOffset = buf_offset + i * spitch;
                slices[i].dstOffset = offset + i * dpitch;
                slices[i].size = width;
            }
        }

        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->buffer->buf.copyBuffer(buf->buffer, dst->buffer, slices);
        फर्काऊ सत्य;
    }
    VK_LOG_DEBUG("STAGING");

    यदि (!sync_staging) {
        // copy was not handled caller needs to fall back
        फर्काऊ असत्य;
    }

    // Staging buffer required
    अचर माप_प्रकार staging_size = width * height;
    ggml_vk_ensure_sync_staging_buffer(dst->device, staging_size);

    vk_buffer& staging_buffer = dst->device->sync_staging;

    std::vector<vk::BufferCopy> slices(1);
    यदि (width == dpitch) {
        slices[0].srcOffset = 0;
        slices[0].dstOffset = offset;
        slices[0].size = staging_size;
    } अन्यथा {
        slices.resize(height);
        लागि (माप_प्रकार i = 0; i < height; i++) {
            slices[i].srcOffset = i * width;
            slices[i].dstOffset = offset + i * dpitch;
            slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer->buf.copyBuffer(staging_buffer->buffer, dst->buffer, slices);

    यदि (width == spitch) {
        deferred_memcpy((uint8_t *)staging_buffer->ptr, src, staging_size, &subctx->in_memcpys);
    } अन्यथा {
        लागि (माप_प्रकार i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *)staging_buffer->ptr + i * width, (अचर uint8_t *) src + i * spitch, width, &subctx->in_memcpys);
        }
    }
    फर्काऊ सत्य;
}

तार्किक ggml_vk_buffer_write_async(vk_context subctx, vk_buffer& dst, माप_प्रकार offset, अचर शून्य * src, माप_प्रकार size, तार्किक sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    फर्काऊ ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, size, size, size, 1, sync_staging);
}

शून्य ggml_vk_buffer_write_2d(vk_buffer& dst, माप_प्रकार offset, अचर शून्य * src, माप_प्रकार spitch, माप_प्रकार dpitch, माप_प्रकार width, माप_प्रकार height) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << width << ", " << height << ")");
    // Buffer is already mapped
    यदि(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        GGML_ASSERT(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        यदि (width == spitch && width == dpitch) {
            स_नकल((uint8_t *)dst->ptr + offset, src, width * height);
        } अन्यथा {
            लागि (माप_प्रकार i = 0; i < height; i++) {
                स_नकल((uint8_t *)dst->ptr + offset + i * dpitch, (अचर uint8_t *) src + i * spitch, width);
            }
        }
    } अन्यथा {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(dst->device, subctx);
        तार्किक ret = ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, spitch, dpitch, width, height, सत्य);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        लागि (स्वतः& cpy : subctx->in_memcpys) {
            स_नकल(cpy.dst, cpy.src, cpy.n);
        }

        लागि (स्वतः& mset : subctx->memsets) {
            स_रखो(mset.dst, mset.val, mset.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, सत्य, UINT64_MAX), "vk_buffer_write_2d waitForFences", dst->device);
        dst->device->device.resetFences({ dst->device->fence });
        ggml_vk_queue_command_pools_cleanup(dst->device);
    }
}

शून्य ggml_vk_buffer_write(vk_buffer& dst, माप_प्रकार offset, अचर शून्य * src, माप_प्रकार size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_buffer_write_2d(dst, offset, src, size, size, size, 1);
}

तार्किक ggml_vk_buffer_read_2d_async(vk_context subctx, vk_buffer& src, माप_प्रकार offset, शून्य * dst, माप_प्रकार spitch, माप_प्रकार dpitch, माप_प्रकार width, माप_प्रकार height, तार्किक sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << offset << ", width=" << width << ", height=" << height << ")");
    GGML_ASSERT(width > 0);
    GGML_ASSERT(height > 0);
    GGML_ASSERT(src != nullptr);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_buffer buf = nullptr;
    माप_प्रकार buf_offset = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_offset);

    std::vector<vk::BufferCopy> slices(1);
    यदि (width == spitch && width == dpitch) {
        // Only do single write if stride is equal
        slices[0].srcOffset = offset;
        slices[0].dstOffset = buf_offset;
        slices[0].size = width * height;
    } अन्यथा {
        slices.resize(height);
        लागि (माप_प्रकार i = 0; i < height; i++) {
            slices[i].srcOffset = offset + i * spitch;
            slices[i].dstOffset = buf_offset + i * dpitch;
            slices[i].size = width;
        }
    }

    यदि (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->buffer->buf.copyBuffer(src->buffer, buf->buffer, slices);

        फर्काऊ सत्य;
    }
    VK_LOG_DEBUG("STAGING");

    यदि (!sync_staging) {
        // copy was not handled caller needs to fall back
        फर्काऊ असत्य;
    }

    // Fall back to staging buffer
    अचर माप_प्रकार staging_size = width * height;
    ggml_vk_ensure_sync_staging_buffer(src->device, staging_size);

    vk_buffer& staging_buffer = src->device->sync_staging;

    std::vector<vk::BufferCopy> staging_slices(1);
    यदि (width == spitch) {
        staging_slices[0].srcOffset = offset;
        staging_slices[0].dstOffset = 0;
        staging_slices[0].size = staging_size;
    } अन्यथा {
        staging_slices.resize(height);
        लागि (माप_प्रकार i = 0; i < height; i++) {
            staging_slices[i].srcOffset = offset + i * spitch;
            staging_slices[i].dstOffset = i * width;
            staging_slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer->buf.copyBuffer(src->buffer, staging_buffer->buffer, staging_slices);

    यदि (width == dpitch) {
        deferred_memcpy(dst, staging_buffer->ptr, staging_size, &subctx->out_memcpys);
    } अन्यथा {
        लागि (माप_प्रकार i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (अचर uint8_t *) staging_buffer->ptr + i * width, width, &subctx->out_memcpys);
        }
    }
    फर्काऊ सत्य;
}

स्थिर तार्किक ggml_vk_buffer_read_async(vk_context subctx, vk_buffer& src, माप_प्रकार offset, शून्य * dst, माप_प्रकार size, तार्किक sync_staging = असत्य) {
    फर्काऊ ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, size, size, size, 1, sync_staging);
}

शून्य ggml_vk_buffer_read_2d(vk_buffer& src, माप_प्रकार offset, शून्य * dst, माप_प्रकार spitch, माप_प्रकार dpitch, माप_प्रकार width, माप_प्रकार height) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->buffer << ", " << offset << ", " << width << ", " << height << ")");

    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    // AMD UMA: uncached host-visible memory is write-combined, CPU reads are slow
    अचर तार्किक slow_host_read = src->device->vendor_id == VK_VENDOR_ID_AMD &&
                                !(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCached);
    यदि(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible && src->device->uma && !slow_host_read) {
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
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, सत्य, UINT64_MAX),
                 "vk_buffer_read_2d uma waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        यदि (width == spitch && width == dpitch) {
            स_नकल(dst, (अचर uint8_t *) src->ptr + offset, width * height);
        } अन्यथा {
            लागि (माप_प्रकार i = 0; i < height; i++) {
                स_नकल((uint8_t *) dst + i * dpitch, (अचर uint8_t *) src->ptr + offset + i * spitch, width);
            }
        }
    } अन्यथा {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        तार्किक ret = ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, spitch, dpitch, width, height, सत्य);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, सत्य, UINT64_MAX), "vk_buffer_read_2d waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        लागि (स्वतः& cpy : subctx->out_memcpys) {
            स_नकल(cpy.dst, cpy.src, cpy.n);
        }
    }
}

शून्य ggml_vk_buffer_read(vk_buffer& src, माप_प्रकार offset, शून्य * dst, माप_प्रकार size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->buffer << ", " << offset << ", " << size << ")");
    ggml_vk_buffer_read_2d(src, offset, dst, size, size, size, 1);
}

शून्य ggml_vk_buffer_copy_async(vk_context& ctx, vk_buffer& dst, माप_प्रकार dst_offset, vk_buffer& src, माप_प्रकार src_offset, माप_प्रकार size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    VkBufferCopy bc{ src_offset, dst_offset, size };

    vkCmdCopyBuffer(ctx->s->buffer->buf, (VkBuffer)src->buffer, (VkBuffer)dst->buffer, 1, &bc);
}

शून्य ggml_vk_buffer_copy(vk_buffer& dst, माप_प्रकार dst_offset, vk_buffer& src, माप_प्रकार src_offset, माप_प्रकार size) {
    यदि (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_buffer_copy_async(subctx, dst, dst_offset, src, src_offset, size);
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, सत्य, UINT64_MAX), "vk_buffer_copy waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);
    } अन्यथा {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_buffer(src->device, size);

        // Copy to src staging buffer
        ggml_vk_buffer_copy(src->device->sync_staging, 0, src, src_offset, size);
        // Copy to dst buffer
        ggml_vk_buffer_write(dst, dst_offset, src->device->sync_staging->ptr, size);
    }
}

शून्य ggml_vk_buffer_memset_async(vk_context& ctx, vk_buffer& dst, माप_प्रकार offset, uint32_t c, माप_प्रकार size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << offset << ", " << c << ", " << size << ")");

    यदि (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->ptr + offset, c, size, &ctx->memsets);
        फर्काऊ;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->buffer->buf.fillBuffer(dst->buffer, offset, size, c);
}

शून्य ggml_vk_buffer_memset(vk_buffer& dst, माप_प्रकार offset, uint32_t c, माप_प्रकार size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << offset << ", " << c << ", " << size << ")");

    यदि (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        स_रखो((uint8_t*)dst->ptr + offset, c, size);
        फर्काऊ;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue->cmd_pool);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->buffer->buf.fillBuffer(dst->buffer, offset, size, c);
    ggml_vk_ctx_end(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, सत्य, UINT64_MAX), "vk_memset waitForFences", dst->device);
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
    /* .reset           = */ रिक्त,
};

vk_buffer ggml_vk_buffer_from_host_ptr(vk_device & device, शून्य * ptr, माप_प्रकार size) {
    यदि (!device->external_memory_host) {
        फर्काऊ {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    यदि (uptr & (device->min_imported_host_pointer_alignment - 1)) {
        फर्काऊ {};
    }
    यदि (size & (device->min_imported_host_pointer_alignment - 1)) {
        फर्काऊ {};
    }

    अचर vk::MemoryPropertyFlags property_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached;

    vk_buffer buf {};
    यत्न {
        buf = ggml_vk_create_buffer(device, size, { property_flags }, ptr);
    } गृहाण (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed ggml_vk_create_buffer (%s)\n", e.what());
    }

    फर्काऊ buf;
}

