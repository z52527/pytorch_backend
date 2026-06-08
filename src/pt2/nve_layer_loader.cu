// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// nvcc-compiled implementation of the NVE layer loader shim (see nve_layer_loader.h).
// Isolated into its own .cu so the CUDA-heavy nve_loader.hpp (cub, cuda_bf16) is
// never pulled into the g++-compiled backend translation units.

#include "nve_layer_loader.h"

#include <sys/stat.h>

#include <iostream>
#include <string>

#include "python/pynve/torch_bindings/nve_loader.hpp"

namespace triton::backend::pytorch::pt2 {

namespace {
bool FileExistsLocal(const std::string& path)
{
  struct stat st;
  return ::stat(path.c_str(), &st) == 0;
}
}  // namespace

void*
NveLoadLayers(const std::string& package_dir, int device_index)
{
  // Gate: only treat this as an NVE model if metadata.json is present, so the
  // backend stays a no-op for ordinary (non-NVE) PyTorch models.
  if (!FileExistsLocal(package_dir + "/metadata.json")) {
    return nullptr;
  }
  try {
    std::cerr << "[nve-loader] loading NVE layers from " << package_dir
              << " (device " << device_index << ")" << std::endl;
    auto* dir = new nve::LayerDirectory(package_dir, device_index);
    std::cerr << "[nve-loader] loaded " << dir->size()
              << " NVE layer(s) into the registry" << std::endl;
    return static_cast<void*>(dir);
  }
  catch (const std::exception& e) {
    std::cerr << "[nve-loader] FAILED to load NVE layers from " << package_dir
              << ": " << e.what() << std::endl;
    return nullptr;
  }
}

void
NveFreeLayers(void* handle)
{
  delete static_cast<nve::LayerDirectory*>(handle);
}

std::size_t
NveLayerCount(void* handle)
{
  if (handle == nullptr) {
    return 0;
  }
  return static_cast<nve::LayerDirectory*>(handle)->size();
}

}  // namespace triton::backend::pytorch::pt2
