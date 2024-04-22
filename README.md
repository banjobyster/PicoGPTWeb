# PicoGPT Web



https://github.com/banjobyster/PicoGPTWeb/assets/77842809/1641b55d-72d8-4fd6-b12c-3ec0a70b322b



PicoGPT Web is a web version of PicoGPT by [Jay Mody](https://github.com/jaymody), inspired by the videos of 3Blue1Brown on GPT and built to create a GPT-2 decoder stack. The project was initiated to gain a hands-on understanding of GPT and explore the usage of WebGPU.

## Acknowledgements

- **PicoGPT**: The original repository for PicoGPT can be found [here](https://github.com/jaymody/picoGPT/).
- **Blog**: Reference was taken from [this blog](https://jaykmody.com/blog/gpt-from-scratch/) which provided valuable insights into building GPT from scratch.
- **3Blue1Brown Videos**: The videos by 3Blue1Brown were instrumental in understanding the concepts behind GPT.

## Project Structure

The project structure consists of several files and folders:

### Components

The `components/` directory contains the following React components:

1. **ErrorDisplayer.tsx**: Component for displaying errors.
2. **InferenceWindow.tsx**: Interface for generating text.
3. **ParamsLoadComponent.tsx**: Component for downloading and loading the model into memory, managing weights in browser storage.
4. **TransformerVisualizer.tsx**: Visualization of transformer block and top-K tokens.

### Common

The `common/` directory includes common functionalities:

- **bpeTokenizer.ts**: BPT Tokenizer.
- **browserIndexedDB.ts**: Functions for IndexedDB storage.
- **gpuUtils.ts**: General Purpose Function for WebGPU.
- **matrixUtils.ts**: Utility function for matrix multiplication operation on WebGPU.
- **paramsExtractor.ts**: Functions for downloading model and extracting parameters.

### Model124M

The `model124M/` directory contains files related to the model:

- **bytes_to_unicode.ts**: Unicode conversion functions.
- **encoder.ts**: Encoder for the model.
- **generate.ts**: GPT-2 generation architecture.
- **hparams.ts**: Hyperparameters for the model.
- **types.d.ts**: Type declarations.
- **vocab.ts**: Vocabulary for tokenization.


## Technologies Used

- **Vite + React**: The project is built using Vite and React for a fast and efficient development experience.
- **WebGPU**: Utilized for enhanced performance and parallel processing in matrix multiplication.
- **Fallback Mechanism**: Includes a fallback code to CPU execution in case WebGPU is not supported. However, browser storage limitations may lead to failures, hence Chrome is recommended for optimal performance.

## Usage

Ensure to use Chrome for optimal performance due to browser storage limitations and full WebGPU support. 
