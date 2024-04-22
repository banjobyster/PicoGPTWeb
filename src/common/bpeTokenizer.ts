import bpe_data from '../model124M/vocab';
import { encoder, decoder } from '../model124M/encoder';
import { byte_encoder, byte_decoder } from '../model124M/bytes_to_unicode';

interface Encoder {
  [key: string]: number;
}

interface Decoder {
  [key: string]: string;
}

interface ByteEncoder {
  [key: number]: string;
}

interface ByteDecoder {
  [key: string]: string;
}

const INF = 1e9;

// Tokenizer class for Byte Pair Encoding (Ref: https://github.com/jaymody/picoGPT/blob/main/encoder.py)
class BPETokenizer {
  private cache: { [key: string]: string };
  private bpeRanks: { [key: string]: number };
  private encoder: Encoder;
  private decoder: Decoder;
  private byte_encoder: ByteEncoder;
  private byte_decoder: ByteDecoder;

  constructor(
    bpeData: string,
    encoder: Encoder,
    decoder: Decoder,
    byte_encoder: ByteEncoder,
    byte_decoder: ByteDecoder
  ) {
    this.cache = {};
    const bpeMerges = bpeData
      .split('\n')
      .slice(1, -1)
      .map((mergeStr) => mergeStr.split(' '));
    this.bpeRanks = Object.fromEntries(bpeMerges.map((merge, index) => [merge.join(' '), index]));
    this.encoder = encoder;
    this.decoder = decoder;
    this.byte_encoder = byte_encoder;
    this.byte_decoder = byte_decoder;
  }

  getPairs(word: string[]): Set<string[]> {
    const pairs = new Set<string[]>();
    let prevChar = word[0];
    for (let i = 1; i < word.length; i++) {
      pairs.add([prevChar, word[i]]);
      prevChar = word[i];
    }
    return pairs;
  }

  bpe(token: string): string {
    if (this.cache[token]) return this.cache[token];
    let word = [...token];
    let pairs = this.getPairs(word);

    if (pairs.size === 0) return token;

    let MAX_ITER = 1000;
    while (true && --MAX_ITER > 0) {
      const bigram = Array.from(pairs).reduce((min, pair) =>
        (this.bpeRanks[pair.join(' ')] ?? INF) < (this.bpeRanks[min.join(' ')] ?? INF) ? pair : min
      );
      const checkBigram = bigram.join(' ');
      if (!(checkBigram in this.bpeRanks)) break;
      const [first, second] = bigram;
      const newWord: string[] = [];
      let i = 0;
      while (i < word.length) {
        const j = word.indexOf(first, i);
        if (j === -1) {
          newWord.push(...word.slice(i));
          break;
        }
        newWord.push(...word.slice(i, j));
        i = j;

        if (word[i] === first && i < word.length - 1 && word[i + 1] === second) {
          newWord.push(first + second);
          i += 2;
        } else {
          newWord.push(word[i]);
          i++;
        }
      }
      word = newWord;
      if (word.length === 1) break;
      else pairs = this.getPairs(word);
    }
    const tokens = word.join(' ');
    this.cache[token] = tokens;
    return tokens;
  }

  public encode(text: string): number[] {
    const pat = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
    const bpeTokens: number[] = [];
    let matches;
    while ((matches = pat.exec(text)) !== null) {
      const token = Array.from(matches[0], (b) => this.byte_encoder[b.charCodeAt(0)]).join('');
      const bpeToken = this.bpe(token)
        .split(' ')
        .map((t: string) => this.encoder[t]);
      bpeTokens.push(...bpeToken);
    }
    return bpeTokens;
  }

  public decode(tokens: number[]): string {
    const text = tokens.map((token) => this.decoder[token]).join('');
    const byteArray = new Uint8Array(text.length);
    for (let i = 0; i < text.length; i++) {
      byteArray[i] = Number(this.byte_decoder[text[i]]);
    }
    const decodedText = new TextDecoder('utf-8', { fatal: false, ignoreBOM: true }).decode(
      byteArray
    );
    return decodedText;
  }
}

export const tokenizer = new BPETokenizer(bpe_data, encoder, decoder, byte_encoder, byte_decoder);
