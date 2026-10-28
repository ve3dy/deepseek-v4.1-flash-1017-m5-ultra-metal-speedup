// Keeps `groups` threadgroups of the GPU busy with ALU work (no memory traffic) for `seconds`, so its clocks stay up.
//   swiftc -O tools/heater.swift -o /tmp/heater && /tmp/heater SECONDS [GROUPS]
import Foundation
import Metal

let seconds = Double(CommandLine.arguments.dropFirst().first ?? "60")!
let groups = Int(CommandLine.arguments.dropFirst(2).first ?? "1")!
let source = """
#include <metal_stdlib>
using namespace metal;
kernel void heat(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint t [[thread_position_in_grid]]) {
  float a = float(t) * 1e-7f, b = 1.0001f;
  for (uint i = 0; i < iters; i++) { a = fma(a, b, 1e-7f); b = fma(b, 0.9999f, 1e-7f); }
  if (a == 12345.0f) out[t] = b;
}
"""
let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(source: source, options: nil)
let p = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "heat")!)
let out = dev.makeBuffer(length: 4 * 32 * groups, options: .storageModeShared)!
let q = dev.makeCommandQueue()!
var iters: UInt32 = 2_000_000
let end = Date().addingTimeInterval(seconds)
var last: MTLCommandBuffer?
while Date() < end {
  let cb = q.makeCommandBuffer()!
  let e = cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(p)
  e.setBuffer(out, offset: 0, index: 0)
  e.setBytes(&iters, length: 4, index: 1)
  e.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
  e.endEncoding()
  cb.commit()
  last?.waitUntilCompleted()
  last = cb
}
last?.waitUntilCompleted()
