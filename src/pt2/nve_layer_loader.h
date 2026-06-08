// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVE embedding-weight loader shim for the PyTorch AOTI backend.
//
// HSTU-style AOTI models call `nve_ops::embedding_lookup(keys, layer_id)`, which
// resolves the embedding table from the process-global NVELayerRegistry at
// execute time. The embedding weights live OUTSIDE model.pt2 (in
// <model_dir>/metadata.json + <model_dir>/weights/*.nve), so loading the .pt2
// does NOT load them — something must explicitly load + register them, exactly
// like the C++ inference demo does with `nve::LayerDirectory(package_path, dev)`.
//
// This header exposes a PLAIN C++ interface so it can be included by the
// g++-compiled backend translation units (model_state.cc). The real
// implementation lives in nve_layer_loader.cu, compiled with nvcc, because
// nve_loader.hpp pulls in CUDA-heavy headers (cub, cuda_bf16) that g++ can't build.

#pragma once

#include <cstddef>
#include <string>

namespace triton::backend::pytorch::pt2 {

// Load NVE layers from `package_dir` (the model version dir holding
// metadata.json + weights/) into the process-global NVELayerRegistry.
//
// Returns an opaque handle the caller MUST keep alive for the model's lifetime:
// the registry bindings are released when the handle is freed (NVE warns that if
// the layers are destroyed, the custom ops fail to find them).
//
// Returns nullptr (safe no-op) when `package_dir` has no metadata.json — i.e. this
// is NOT an NVE model — so the backend stays correct for ordinary PyTorch models.
// Returns nullptr on load failure (logged).
void* NveLoadLayers(const std::string& package_dir, int device_index);

// Release a handle returned by NveLoadLayers (no-op on nullptr).
void NveFreeLayers(void* handle);

// Number of layers held by the handle (0 on nullptr).
std::size_t NveLayerCount(void* handle);

}  // namespace triton::backend::pytorch::pt2
