import { generalPurposeGpuFunction, getGpu } from './gpuUtils';

const convertMatrixTo1D = (matrix: number[][]) => {
  const rows = matrix.length;
  const cols = matrix[0].length;

  if (rows * cols > 5000000) {
    // If small, then later on converting 2D to buffer is slower
    // But if large, then that extra time can be ignored
    // @ts-expect-error - TS cannot infer that matrix is being spread
    return [].concat(...matrix);
  }

  const output = [];
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[0].length; j++) {
      output.push(matrix[i][j]);
    }
  }
  return output;
};

const convert1DToMatrix = (
  array: number[] | Float32Array | undefined,
  rows: number,
  cols: number
) => {
  if (array === undefined) {
    console.warn('Array is undefined.');
    return;
  }
  const output = [];
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      row.push(array[i * cols + j]);
    }
    output.push(row);
  }
  return output;
};

interface matrixMultiplicationProps {
  matA: number[][];
  matB: number[][];
}

const matrixMultiplication = async (props: matrixMultiplicationProps) => {
  let gpuPresent = true;
  const { matA, matB } = props;
  const rowsA = matA.length;
  const colsA = matA[0].length;
  const rowsB = matB.length;
  const colsB = matB[0].length;

  if (colsA !== rowsB) {
    throw Error('Matrix multiplication not possible.');
  }

  try {
    await getGpu();
  } catch (error) {
    gpuPresent = false;
  }

  if (rowsA * colsB < 4000 || gpuPresent === false) {
    const result = Array.from({ length: rowsA }, () => Array.from({ length: colsB }, () => 0));

    for (let i = 0; i < rowsA; i++) {
      for (let j = 0; j < colsB; j++) {
        for (let k = 0; k < colsA; k++) {
          result[i][j] += matA[i][k] * matB[k][j];
        }
      }
    }

    return result;
  }
  const rowMatrixA = convertMatrixTo1D(matA);
  const rowMatrixB = convertMatrixTo1D(matB);

  const moduleFunction = (WORKGROUP_SIZE: number) => {
    return `
            @group(0) @binding(0)
            var<storage, read> input0: array<f32>;
            @group(0) @binding(1)
            var<storage, read> input1: array<f32>;
            @group(0) @binding(2)
            var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(${WORKGROUP_SIZE})
            fn main(
                @builtin(global_invocation_id)
                global_id : vec3<u32>
            ) {
                if(global_id.x >= arrayLength(&output)) {
                    return;
                }
                let row = global_id.x / ${colsB};
                let col = global_id.x % ${colsB};
                var sum = 0.0;
                for (var i = 0u; i < ${colsA}; i = i + 1u) {
                    sum = sum + input0[row * ${colsA} + i] * input1[i * ${colsB} + col];
                }
                output[global_id.x] = sum;
            }
        `;
  };

  const output = await generalPurposeGpuFunction({
    inputs: [rowMatrixA, rowMatrixB],
    moduleFunction,
    workElements: rowsA * colsB
  });

  return convert1DToMatrix(output, rowsA, colsB);
};

export { matrixMultiplication };
