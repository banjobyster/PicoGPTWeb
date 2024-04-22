let GPU_DEVICE: GPUDevice | null = null;

// Function to get GPU Device of WebGPU
const getGpu = async () => {
  // Return device if already exists
  if (GPU_DEVICE) return GPU_DEVICE;

  if (!navigator.gpu) throw Error('WebGPU not supported. Animations Stopped!');

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw Error('Couldn’t request WebGPU adapter. Animations Stopped!');

  const requestedDevice = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: 154389504
    }
  });
  if (!requestedDevice) throw Error('Couldn’t request WebGPU logical device. Animations Stopped!');

  // Set device in store and return device
  GPU_DEVICE = requestedDevice;
  return requestedDevice;
};

interface GeneralPurposeGPUFunctionProps {
  inputs: number[][];
  moduleFunction: (WORKGROUP_SIZE: number) => string;
  workElements: number;
}

// General purpose function to perform computation on GPU
const generalPurposeGpuFunction = async (props: GeneralPurposeGPUFunctionProps) => {
  const { inputs, moduleFunction, workElements } = props;
  const WORKGROUP_SIZE = 64;

  // Check if inputs are provided
  if (inputs.length === 0) {
    console.warn('No inputs provided.');
    return;
  }

  // Check if device is provided, else get device
  let logicalDevice = GPU_DEVICE;
  if (GPU_DEVICE === null) {
    logicalDevice = await getGpu();
  }

  // Create shader module
  const module = logicalDevice!.createShaderModule({
    code: moduleFunction(WORKGROUP_SIZE)
  });

  // Create input buffers, each with binding equal to its index
  const inputBuffers = inputs.map((input) => {
    const BUFFER_SIZE = input.length * Float32Array.BYTES_PER_ELEMENT;
    const buffer = logicalDevice!.createBuffer({
      size: BUFFER_SIZE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    logicalDevice!.queue.writeBuffer(buffer, 0, new Float32Array(input));
    return buffer;
  });

  // Create output buffer
  const outputBuffer = logicalDevice!.createBuffer({
    size: workElements * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  // Create staging buffer to read from output buffer
  const stagingBuffer = logicalDevice!.createBuffer({
    size: workElements * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  });

  // Create bind group layout with entries for each input buffer and one for the output buffer
  const inputBuffersBindGroupEntries = inputBuffers.map((_, index) => ({
    binding: index,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: 'read-only-storage' }
  }));

  const bindGroupLayout = logicalDevice!.createBindGroupLayout({
    // @ts-expect-error - TS cannot infer that inputBuffersBindGroupEntries is being spread
    entries: [
      ...inputBuffersBindGroupEntries,
      {
        binding: inputBuffers.length,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      }
    ]
  });

  // Create bind group with entries for each input buffer and one for the output buffer
  const bindGroupEntries = inputBuffers.map((buffer, index) => ({
    binding: index,
    resource: { buffer }
  }));

  const bindGroup = logicalDevice!.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      ...bindGroupEntries,
      {
        binding: inputBuffers.length,
        resource: { buffer: outputBuffer }
      }
    ]
  });

  // Create compute pipeline
  const pipeline = logicalDevice!.createComputePipeline({
    compute: {
      module,
      entryPoint: 'main'
    },
    layout: logicalDevice!.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    })
  });

  // Create command encoder and set pipeline and bind group and dispatch workgroups
  const commandEncoder = logicalDevice!.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(Math.ceil(workElements / WORKGROUP_SIZE));
  passEncoder.end();

  // Copy output buffer to staging buffer
  commandEncoder.copyBufferToBuffer(
    outputBuffer,
    0,
    stagingBuffer,
    0,
    workElements * Float32Array.BYTES_PER_ELEMENT
  );

  // Submit command encoder to device queue
  logicalDevice!.queue.submit([commandEncoder.finish()]);

  // Map staging buffer and get data and copy the data to new array buffer
  await stagingBuffer.mapAsync(GPUMapMode.READ, 0, workElements * Float32Array.BYTES_PER_ELEMENT);
  const copyArrayBuffer = stagingBuffer.getMappedRange(
    0,
    workElements * Float32Array.BYTES_PER_ELEMENT
  );

  // Copy the data to new array buffer and unmap the staging buffer
  const data = copyArrayBuffer.slice(0);
  stagingBuffer.unmap();
  return new Float32Array(data);
};

export { generalPurposeGpuFunction, getGpu };
