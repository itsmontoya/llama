// This library is loaded only in diagnostic processes. It observes the
// released implementation's public Metal calls without changing its module.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdio.h>
#include <time.h>

static const NSUInteger maxEncoders = 512;
static char commandKey, encoderKey, pipelineKey;
static _Atomic uint64_t nextCommand = 0;
static BOOL sampleStages;
static id<MTLCounterSet> timestampSet;

static uint64_t nowNS(void) {
	return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

// Field order is bound to v0.11.2's NNMetalMatMulDimensions. Inspect lengths
// and the pipeline name before decoding bytes supplied by the caller.
typedef struct {
	uint32_t rows, inputWidth, weightRows, weightCols, resultRows, outputWidth, variant;
} MatMulDimensions;

typedef struct {
	uint64_t liveBuffers, liveBufferBytes, peakBuffers, peakBufferBytes;
	uint64_t liveScopes, peakScopes, createdBuffers, releasedBuffers;
	uint64_t createdScopes, releasedScopes, submittedCommands, completedCommands;
} ResourceSnapshot;

@interface LL04Encoder : NSObject
@property NSString *name;
@property MatMulDimensions dimensions;
@property NSUInteger dispatches;
@end
@implementation LL04Encoder
@end

@interface LL04Command : NSObject
@property uint64_t identifier;
@property id<MTLCounterSampleBuffer> samples;
@property NSMutableArray<LL04Encoder *> *encoders;
@property NSUInteger blits;
@property uint64_t commitNS;
@property MTLTimestamp cpuStart;
@property MTLTimestamp gpuStart;
@property BOOL reported;
@end
@implementation LL04Command
@end

// Install on the concrete object class. If it inherits the selector, add
// an override locally so that unrelated subclasses retain their methods.
static void hook(Class cls, SEL selector, IMP replacement) {
	Method method = class_getInstanceMethod(cls, selector);
	if (!method) {
		fprintf(stderr, "LL04 ERROR missing selector %s\n", sel_getName(selector));
		abort();
	}
	if (!class_addMethod(cls, selector, replacement, method_getTypeEncoding(method))) {
		method_setImplementation(method, replacement);
	}
}

static IMP implementation(Class cls, SEL selector) {
	return class_getMethodImplementation(cls, selector);
}

static void hookEncoder(id<MTLComputeCommandEncoder> encoder) {
	static NSMutableSet *classes;
	@synchronized ([LL04Encoder class]) {
		if (!classes) {
			classes = [NSMutableSet new];
		}
		Class cls = object_getClass(encoder);
		if ([classes containsObject:cls]) {
			return;
		}
		SEL setPipeline = @selector(setComputePipelineState:);
		SEL setBytes = @selector(setBytes:length:atIndex:);
		SEL groups = @selector(dispatchThreadgroups:threadsPerThreadgroup:);
		SEL threads = @selector(dispatchThreads:threadsPerThreadgroup:);
		void (*originalPipeline)(id, SEL, id) = (void *)implementation(cls, setPipeline);
		void (*originalBytes)(id, SEL, const void *, NSUInteger, NSUInteger) = (void *)implementation(cls, setBytes);
		void (*originalGroups)(id, SEL, MTLSize, MTLSize) = (void *)implementation(cls, groups);
		void (*originalThreads)(id, SEL, MTLSize, MTLSize) = (void *)implementation(cls, threads);
		hook(cls, setPipeline, imp_implementationWithBlock(^(id object, id pipeline) {
			LL04Encoder *event = objc_getAssociatedObject(object, &encoderKey);
			event.name = objc_getAssociatedObject(pipeline, &pipelineKey);
			originalPipeline(object, setPipeline, pipeline);
		}));
		hook(cls, setBytes, imp_implementationWithBlock(^(id object, const void *bytes, NSUInteger length, NSUInteger index) {
			LL04Encoder *event = objc_getAssociatedObject(object, &encoderKey);
			if (index == 3 && length == sizeof(MatMulDimensions) && [event.name hasPrefix:@"nn_matmul"]) {
				MatMulDimensions dimensions;
				memcpy(&dimensions, bytes, sizeof(dimensions));
				event.dimensions = dimensions;
			}
			originalBytes(object, setBytes, bytes, length, index);
		}));
		hook(cls, groups, imp_implementationWithBlock(^(id object, MTLSize grid, MTLSize group) {
			LL04Encoder *event = objc_getAssociatedObject(object, &encoderKey);
			event.dispatches++;
			originalGroups(object, groups, grid, group);
		}));
		hook(cls, threads, imp_implementationWithBlock(^(id object, MTLSize grid, MTLSize group) {
			LL04Encoder *event = objc_getAssociatedObject(object, &encoderKey);
			event.dispatches++;
			originalThreads(object, threads, grid, group);
		}));
		[classes addObject:cls];
	}
}

static void reportCommand(id<MTLCommandBuffer> command, LL04Command *event, uint64_t waitNS) {
	NSUInteger count = event.encoders.count;
	NSUInteger dispatches = 0;
	for (LL04Encoder *encoder in event.encoders) {
		dispatches += encoder.dispatches;
	}
	double gpuStart = command.GPUStartTime;
	double gpuEnd = command.GPUEndTime;
	fprintf(stderr, "LL04 COMMAND %llu %lu %lu %lu %llu %llu %.0f %.0f %lu %lu\n",
		event.identifier, count, event.blits, dispatches, event.commitNS, waitNS,
		gpuStart * 1e9, gpuEnd * 1e9, command.status, command.device.currentAllocatedSize);
	void (*snapshot)(ResourceSnapshot *) = dlsym(RTLD_MAIN_ONLY, "nn_metal_resource_snapshot");
	if (snapshot) {
		ResourceSnapshot resources;
		snapshot(&resources);
		fprintf(stderr, "LL04 RESOURCES %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n",
			event.identifier, resources.liveBuffers, resources.liveBufferBytes,
			resources.peakBuffers, resources.peakBufferBytes, resources.liveScopes, resources.peakScopes,
			resources.createdBuffers, resources.releasedBuffers, resources.createdScopes, resources.releasedScopes,
			resources.submittedCommands, resources.completedCommands);
	}
	if (!event.samples || !count) {
		return;
	}
	MTLTimestamp cpuEnd = 0, gpuEndTimestamp = 0;
	[command.device sampleTimestamps:&cpuEnd gpuTimestamp:&gpuEndTimestamp];
	fprintf(stderr, "LL04 CALIBRATION %llu %llu %llu %llu %llu\n",
		event.identifier, event.cpuStart, cpuEnd, event.gpuStart, gpuEndTimestamp);
	NSData *resolved = [event.samples resolveCounterRange:NSMakeRange(0, 2 * count)];
	if (resolved.length != 2 * count * sizeof(MTLCounterResultTimestamp)) {
		fprintf(stderr, "LL04 ERROR command %llu counter resolve returned %lu bytes\n",
			event.identifier, resolved.length);
		return;
	}
	const MTLCounterResultTimestamp *times = resolved.bytes;
	for (NSUInteger i = 0; i < count; i++) {
		LL04Encoder *encoder = event.encoders[i];
		MatMulDimensions dimensions = encoder.dimensions;
		fprintf(stderr, "LL04 ENCODER %llu %lu %s %u %u %u %u %llu %llu\n",
			event.identifier, i, encoder.name.UTF8String ?: "unknown",
			dimensions.rows, dimensions.inputWidth, dimensions.outputWidth, dimensions.variant,
			times[2*i].timestamp, times[2*i+1].timestamp);
	}
}

static void hookCommand(id<MTLCommandBuffer> command) {
	static NSMutableSet *classes;
	@synchronized ([LL04Command class]) {
		if (!classes) {
			classes = [NSMutableSet new];
		}
		Class cls = object_getClass(command);
		if ([classes containsObject:cls]) {
			return;
		}
		SEL compute = @selector(computeCommandEncoder);
		SEL blit = @selector(blitCommandEncoder);
		SEL commit = @selector(commit);
		SEL wait = @selector(waitUntilCompleted);
		id (*originalCompute)(id, SEL) = (void *)implementation(cls, compute);
		id (*originalBlit)(id, SEL) = (void *)implementation(cls, blit);
		void (*originalCommit)(id, SEL) = (void *)implementation(cls, commit);
		void (*originalWait)(id, SEL) = (void *)implementation(cls, wait);
		hook(cls, compute, imp_implementationWithBlock(^id(id<MTLCommandBuffer> object) {
			LL04Command *event = objc_getAssociatedObject(object, &commandKey);
			if (!event) {
				return originalCompute(object, compute);
			}
			NSUInteger index = event.encoders.count;
			id<MTLComputeCommandEncoder> encoder;
			if (event.samples && index < maxEncoders) {
				MTLComputePassDescriptor *pass = [MTLComputePassDescriptor computePassDescriptor];
				MTLComputePassSampleBufferAttachmentDescriptor *attachment = pass.sampleBufferAttachments[0];
				attachment.sampleBuffer = event.samples;
				attachment.startOfEncoderSampleIndex = 2 * index;
				attachment.endOfEncoderSampleIndex = 2 * index + 1;
				encoder = [object computeCommandEncoderWithDescriptor:pass];
			} else {
				encoder = originalCompute(object, compute);
			}
			if (encoder) {
				hookEncoder(encoder);
				LL04Encoder *entry = [LL04Encoder new];
				[event.encoders addObject:entry];
				objc_setAssociatedObject(encoder, &encoderKey, entry, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
			}
			return encoder;
		}));
		hook(cls, blit, imp_implementationWithBlock(^id(id object) {
			id encoder = originalBlit(object, blit);
			LL04Command *event = objc_getAssociatedObject(object, &commandKey);
			if (encoder) {
				event.blits++;
			}
			return encoder;
		}));
		hook(cls, commit, imp_implementationWithBlock(^(id<MTLCommandBuffer> object) {
			LL04Command *event = objc_getAssociatedObject(object, &commandKey);
			if (event.samples) {
				MTLTimestamp cpuStart = 0, gpuStart = 0;
				[object.device sampleTimestamps:&cpuStart gpuTimestamp:&gpuStart];
				event.cpuStart = cpuStart;
				event.gpuStart = gpuStart;
			}
			uint64_t start = nowNS();
			originalCommit(object, commit);
			event.commitNS = nowNS() - start;
		}));
		hook(cls, wait, imp_implementationWithBlock(^(id<MTLCommandBuffer> object) {
			LL04Command *event = objc_getAssociatedObject(object, &commandKey);
			uint64_t start = nowNS();
			originalWait(object, wait);
			uint64_t elapsed = nowNS() - start;
			if (event && !event.reported) {
				event.reported = YES;
				reportCommand(object, event, elapsed);
			}
		}));
		[classes addObject:cls];
	}
}

static void hookQueue(id<MTLCommandQueue> queue) {
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		Class cls = object_getClass(queue);
		SEL selector = @selector(commandBuffer);
		id (*original)(id, SEL) = (void *)implementation(cls, selector);
		hook(cls, selector, imp_implementationWithBlock(^id(id<MTLCommandQueue> object) {
			id<MTLCommandBuffer> command = original(object, selector);
			if (command) {
				hookCommand(command);
				LL04Command *event = [LL04Command new];
				event.identifier = atomic_fetch_add(&nextCommand, 1);
				event.encoders = [NSMutableArray arrayWithCapacity:maxEncoders];
				if (sampleStages) {
					MTLCounterSampleBufferDescriptor *description = [MTLCounterSampleBufferDescriptor new];
					description.label = @"LL04 compute stages";
					description.counterSet = timestampSet;
					description.storageMode = MTLStorageModeShared;
					description.sampleCount = 2 * maxEncoders;
					NSError *error = nil;
					event.samples = [object.device newCounterSampleBufferWithDescriptor:description error:&error];
					if (!event.samples) {
						fprintf(stderr, "LL04 ERROR counter sample buffer: %s\n", error.description.UTF8String);
					}
				}
				objc_setAssociatedObject(command, &commandKey, event, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
			}
			return command;
		}));
	});
}

typedef id (*NewLibrary)(id, SEL, NSString *, MTLCompileOptions *, NSError **) NS_RETURNS_RETAINED;
typedef id (*NewPipeline)(id, SEL, id, NSError **) NS_RETURNS_RETAINED;
typedef id (*NewQueue)(id, SEL) NS_RETURNS_RETAINED;
static NewLibrary originalLibrary;
static NewPipeline originalPipeline;
static NewQueue originalQueue;

static id profileLibrary(id object, SEL selector, NSString *source,
	MTLCompileOptions *options, NSError **error) NS_RETURNS_RETAINED;
static id profileLibrary(id object, SEL selector, NSString *source,
	MTLCompileOptions *options, NSError **error) {
	uint64_t begin = nowNS();
	id value = originalLibrary(object, selector, source, options, error);
	fprintf(stderr, "LL04 LIBRARY %lu %llu %d\n", source.length, nowNS() - begin, value != nil);
	return value;
}

static id profilePipeline(id object, SEL selector, id<MTLFunction> function,
	NSError **error) NS_RETURNS_RETAINED;
static id profilePipeline(id object, SEL selector, id<MTLFunction> function,
	NSError **error) {
	uint64_t begin = nowNS();
	id value = originalPipeline(object, selector, function, error);
	fprintf(stderr, "LL04 PIPELINE %s %llu %d\n", function.name.UTF8String, nowNS() - begin, value != nil);
	if (value) {
		objc_setAssociatedObject(value, &pipelineKey, function.name, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
	}
	return value;
}

static id profileQueue(id object, SEL selector) NS_RETURNS_RETAINED;
static id profileQueue(id object, SEL selector) {
	uint64_t begin = nowNS();
	id<MTLCommandQueue> value = originalQueue(object, selector);
	fprintf(stderr, "LL04 QUEUE %llu %d\n", nowNS() - begin, value != nil);
	if (value) {
		hookQueue(value);
	}
	return value;
}

static id<MTLDevice> createDevice(void) NS_RETURNS_RETAINED;
static id<MTLDevice> createDevice(void) {
	uint64_t start = nowNS();
	id<MTLDevice> device = MTLCreateSystemDefaultDevice();
	uint64_t elapsed = nowNS() - start;
	if (!device) {
		return nil;
	}
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		BOOL stage = [device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
		BOOL dispatch = [device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary];
		for (id<MTLCounterSet> counters in device.counterSets) {
			if ([counters.name isEqualToString:MTLCommonCounterSetTimestamp]) {
				timestampSet = counters;
			}
		}
		sampleStages = stage && timestampSet && getenv("LLAMA_METAL_SAMPLE") != NULL;
		fprintf(stderr, "LL04 DEVICE %s %llu %d %d %d %d %llu\n", device.name.UTF8String,
			elapsed, stage, dispatch, timestampSet != nil, sampleStages, device.recommendedMaxWorkingSetSize);
		Class cls = object_getClass(device);
		SEL library = @selector(newLibraryWithSource:options:error:);
		SEL pipeline = @selector(newComputePipelineStateWithFunction:error:);
		SEL queue = @selector(newCommandQueue);
		originalLibrary = (NewLibrary)implementation(cls, library);
		originalPipeline = (NewPipeline)implementation(cls, pipeline);
		originalQueue = (NewQueue)implementation(cls, queue);
		hook(cls, library, (IMP)profileLibrary);
		hook(cls, pipeline, (IMP)profilePipeline);
		hook(cls, queue, (IMP)profileQueue);
	});
	return device;
}

__attribute__((used, section("__DATA,__interpose")))
static const struct {
	const void *replacement;
	const void *original;
} deviceInterpose = {
	(const void *)createDevice,
	(const void *)MTLCreateSystemDefaultDevice,
};
