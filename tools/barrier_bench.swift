// What a dependent dispatch costs on this GPU, by synchronization method.
//   swiftc -O tools/barrier_bench.swift -o /tmp/barrier_bench && /tmp/barrier_bench
// Two chains of N dispatches: `tiny` (one thread bumps a counter) and `stream` (the whole GPU reads a fresh 16 MB
// slice, then bumps the counter). Each method's ms for the chain, µs a step, and whether the counter reached N
// (a step saw the one before's write).
import Foundation
import Metal

let source = """
#include <metal_stdlib>
using namespace metal;
kernel void tiny(device uint* a [[buffer(0)]], uint t [[thread_position_in_grid]]) {
  if (t == 0) a[0] = a[0] + 1;
}
kernel void stream(device const uint4* src [[buffer(0)]], device uint* a [[buffer(1)]],
                   constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]],
                   uint gsz [[threads_per_grid]]) {
  uint4 acc = 0;
  for (uint i = gid; i < n; i += gsz) acc += src[i];
  if (acc.x == 0xdeadbeefu) a[1] = acc.y;
  if (gid == 0) a[0] = a[0] + 1;
}
"""

let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(source: source, options: nil)
let tiny = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "tiny")!)
let stream = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "stream")!)
let counter = dev.makeBuffer(length: 64, options: .storageModeShared)!
let sliceBytes = 16 << 20
let slices = 32
let big = dev.makeBuffer(length: sliceBytes * slices, options: .storageModePrivate)!
var sliceWords = UInt32(sliceBytes / 16)
let n = 400
let streamGrid = MTLSize(width: 4096 * 256, height: 1, depth: 1)
let group = MTLSize(width: 256, height: 1, depth: 1)

enum Kind { case tiny, stream }

func reset() { counter.contents().storeBytes(of: UInt32(0), as: UInt32.self) }
func count() -> UInt32 { counter.contents().load(as: UInt32.self) }

// The classic API: a compute encoder, `sync` between dispatches.
func legacy(_ kind: Kind, _ dispatch: MTLDispatchType, _ sync: (MTLComputeCommandEncoder) -> Void) -> (Double, UInt32) {
  let q = dev.makeCommandQueue()!
  var best = Double.infinity
  var seen: UInt32 = 0
  for _ in 0..<5 {
    reset()
    let cb = q.makeCommandBufferWithUnretainedReferences()!
    let e = cb.makeComputeCommandEncoder(dispatchType: dispatch)!
    for i in 0..<n {
      if i > 0 { sync(e) }
      switch kind {
      case .tiny:
        e.setComputePipelineState(tiny)
        e.setBuffer(counter, offset: 0, index: 0)
        e.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
      case .stream:
        e.setComputePipelineState(stream)
        e.setBuffer(big, offset: (i % slices) * sliceBytes, index: 0)
        e.setBuffer(counter, offset: 0, index: 1)
        e.setBytes(&sliceWords, length: 4, index: 2)
        e.dispatchThreads(streamGrid, threadsPerThreadgroup: group)
      }
    }
    e.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    best = min(best, cb.gpuEndTime - cb.gpuStartTime)
    seen = count()
  }
  return (best * 1e3, seen)
}

// Metal 4: an argument table, a residency set, an intra-pass barrier with `vis` (nil: none at all).
func metal4(_ kind: Kind, _ vis: MTL4VisibilityOptions?) -> (Double, UInt32) {
  let q = dev.makeMTL4CommandQueue()!
  let alloc = dev.makeCommandAllocator()!
  let rd = MTLResidencySetDescriptor()
  let rs = try! dev.makeResidencySet(descriptor: rd)
  rs.addAllocation(counter)
  rs.addAllocation(big)
  rs.commit()
  q.addResidencySet(rs)
  let td = MTL4ArgumentTableDescriptor()
  td.maxBufferBindCount = 3
  let table = try! dev.makeArgumentTable(descriptor: td)
  let args = dev.makeBuffer(length: 16, options: .storageModeShared)!
  args.contents().storeBytes(of: sliceWords, as: UInt32.self)
  rs.addAllocation(args)
  rs.commit()
  let event = dev.makeSharedEvent()!
  var value: UInt64 = 0
  var best = Double.infinity
  var seen: UInt32 = 0
  for _ in 0..<5 {
    reset()
    alloc.reset()
    let cb = dev.makeCommandBuffer()!
    cb.beginCommandBuffer(allocator: alloc)
    let e = cb.makeComputeCommandEncoder()!
    e.setArgumentTable(table)
    for i in 0..<n {
      if i > 0, let v = vis { e.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: v) }
      switch kind {
      case .tiny:
        e.setComputePipelineState(tiny)
        table.setAddress(counter.gpuAddress, index: 0)
        e.dispatchThreads(threadsPerGrid: MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
      case .stream:
        e.setComputePipelineState(stream)
        table.setAddress(big.gpuAddress + UInt64((i % slices) * sliceBytes), index: 0)
        table.setAddress(counter.gpuAddress, index: 1)
        table.setAddress(args.gpuAddress, index: 2)
        e.dispatchThreads(threadsPerGrid: streamGrid, threadsPerThreadgroup: group)
      }
    }
    e.endEncoding()
    cb.endCommandBuffer()
    value += 1
    let t0 = DispatchTime.now().uptimeNanoseconds
    q.commit([cb])
    q.signalEvent(event, value: value)
    _ = event.wait(untilSignaledValue: value, timeoutMS: 10_000)
    let t1 = DispatchTime.now().uptimeNanoseconds
    best = min(best, Double(t1 - t0) / 1e6)
    seen = count()
  }
  q.removeResidencySet(rs)
  return (best, seen)
}

func report(_ name: String, _ r: (Double, UInt32)) {
  print(String(format: "%-44@ %8.3f ms  %6.2f us a step  counter %u/%d", name as NSString, r.0, r.0 * 1e3 / Double(n), r.1, n))
}

for kind in [Kind.tiny, Kind.stream] {
  let k = kind == .tiny ? "tiny" : "stream 16MB"
  report("\(k): concurrent, no barrier", legacy(kind, .concurrent) { _ in })
  report("\(k): concurrent + memoryBarrier(buffers)", legacy(kind, .concurrent) { $0.memoryBarrier(scope: .buffers) })
  report("\(k): concurrent + memoryBarrier(resources)", legacy(kind, .concurrent) { $0.memoryBarrier(resources: [counter, big]) })
  report("\(k): serial encoder", legacy(kind, .serial) { _ in })
  report("\(k): Metal 4, no barrier", metal4(kind, nil))
  report("\(k): Metal 4 barrier, visibility none", metal4(kind, []))
  report("\(k): Metal 4 barrier, visibility device", metal4(kind, .device))
}
