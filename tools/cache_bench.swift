// How much of a just-read buffer a second read finds in the GPU's caches, by size.
//   swiftc -O tools/cache_bench.swift -o /tmp/cache_bench && /tmp/cache_bench
// For each size: read a cold region (a 4 GB buffer walked so no line repeats), then the same region again in the
// same command buffer behind a barrier; GB/s of each read, from encoder timestamps.
import Foundation
import Metal

let source = """
#include <metal_stdlib>
using namespace metal;
kernel void stream(device const uint4* src [[buffer(0)]], device uint* sink [[buffer(1)]],
                   constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]],
                   uint gsz [[threads_per_grid]]) {
  uint4 acc = 0;
  for (uint i = gid; i < n; i += gsz) acc += src[i];
  if (acc.x == 0xdeadbeefu) sink[0] = acc.y;
}
"""
let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(source: source, options: nil)
let stream = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "stream")!)
let total = 4 << 30
let big = dev.makeBuffer(length: total, options: .storageModePrivate)!
let sink = dev.makeBuffer(length: 64, options: .storageModeShared)!
let q = dev.makeCommandQueue()!
let set = dev.counterSets!.first { $0.name == MTLCommonCounterSet.timestamp.rawValue }!
let d = MTLCounterSampleBufferDescriptor()
d.counterSet = set
d.storageMode = .shared
d.sampleCount = 8
let sb = try! dev.makeCounterSampleBuffer(descriptor: d)
var grid = MTLSize(width: 8192 * 256, height: 1, depth: 1)
var group = MTLSize(width: 256, height: 1, depth: 1)
var offset = 0

func pass(_ cb: MTLCommandBuffer, _ at: Int, _ bytes: Int, _ s: Int) {
  let p = MTLComputePassDescriptor()
  p.dispatchType = .serial
  p.sampleBufferAttachments[0].sampleBuffer = sb
  p.sampleBufferAttachments[0].startOfEncoderSampleIndex = s
  p.sampleBufferAttachments[0].endOfEncoderSampleIndex = s + 1
  let e = cb.makeComputeCommandEncoder(descriptor: p)!
  var words = UInt32(bytes / 16)
  e.setComputePipelineState(stream)
  e.setBuffer(big, offset: at, index: 0)
  e.setBuffer(sink, offset: 0, index: 1)
  e.setBytes(&words, length: 4, index: 2)
  e.dispatchThreads(grid, threadsPerThreadgroup: group)
  e.endEncoding()
}

let shapes: [(Int, Int)] = CommandLine.arguments.count > 1 ? [(256, 256), (512, 256), (1024, 256), (2048, 256), (8192, 256), (128, 1024), (256, 1024), (512, 1024), (1024, 1024)] : [(8192, 256)]
for (tgs, tsize) in shapes {
grid = MTLSize(width: tgs * tsize, height: 1, depth: 1)
group = MTLSize(width: tsize, height: 1, depth: 1)
print("grid \(tgs) x \(tsize)")
for mb in CommandLine.arguments.count > 1 ? [8, 16, 32] : [8, 16, 32, 48, 64, 96, 128, 192, 256] {
  let bytes = mb << 20
  var cold = 0.0, warm = 0.0
  for _ in 0..<3 {
    if offset + bytes > total { offset = 0 }
    let cb = q.makeCommandBuffer()!
    pass(cb, offset, bytes, 0)
    pass(cb, offset, bytes, 2)
    cb.commit()
    cb.waitUntilCompleted()
    offset += bytes
    let data = try! sb.resolveCounterRange(0..<4)!
    let t = data.withUnsafeBytes { Array($0.bindMemory(to: UInt64.self)) }
    cold = max(cold, Double(bytes) / Double(t[1] - t[0]))
    warm = max(warm, Double(bytes) / Double(t[3] - t[2]))
  }
  print(String(format: "%4d MB: first read %6.0f GB/s, again %6.0f GB/s", mb, cold, warm))
}
}
