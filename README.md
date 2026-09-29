# Parallel Bloom Filter (CUDA)

A Bloom filter built twice, once as a sequential CPU version and once as a parallel GPU version in CUDA, so the two can be timed against each other. Written for COP4520 (Project 3).

The program generates a set of random strings, inserts them into the filter, then queries every one of them back. It reports the time each version took, the GPU speedup over the CPU, and the number of false negatives. A correct Bloom filter should have zero false negatives.

## How it works

### Bloom filter

A Bloom filter is a compact, probabilistic set. To insert a string, it is hashed `k` times and each hash picks a slot in an array of `m` slots, which gets set to 1. To query a string, it is hashed the same way. If any of those slots is 0, the string was definitely never inserted. If all are 1, the string was *probably* inserted (a false positive is possible, a false negative is not).

The filter's size and number of hash functions come from the element count `n` and the desired false-positive rate `p`, using the standard [optimal formulas](https://en.wikipedia.org/wiki/Bloom_filter#Optimal_number_of_hash_functions) (see `init_filter`):

```
m = ceil( -n · ln(p) / (ln 2)² )     // number of slots
k = round( (m / n) · ln 2 )          // number of hash functions
```

For simplicity, each slot is stored as a full byte (`uint8_t`) rather than a single bit. Commented-out `set_bit` / `check_bit` macros in the source show how to pack it into bits to use 8× less memory.

### Hashing

Hashes come from **SipHash-2-4**, using the [reference C implementation](https://github.com/veorq/SipHash) by Jean-Philippe Aumasson and Daniel J. Bernstein (public domain, CC0). To get `k` different hashes from one function, the 16-byte key starts at `{1, 0, …, 0}` and after each round is XORed with bytes of the previous hash, so each round hashes the string under a new key.

SipHash exists twice in the file: `siphash` for the host and `device_siphash` (a `__device__` copy) for the GPU. The add and check functions are duplicated the same way.

### Test data

`generate_flattened_string` creates `n` random strings, each 1–20 characters long, drawn from letters, digits and spaces. Instead of `n` separate allocations, all strings are packed end to end into one null-terminated `char` buffer, with a separate `positions` array giving each string's starting offset. This avoids thousands of `malloc` calls and lets the whole set be copied to the GPU in a single `cudaMemcpy`. The random seed is fixed (`srand(1)`), so every run uses the same strings.

### CPU vs. GPU

| | CPU | GPU |
|---|---|---|
| Insert | Loop over all strings, calling `add_to_filter` | `addKernel`: one thread per string |
| Query | Loop over all strings, calling `check_filter` | `checkKernel`: one thread per string |
| Miss count | Plain counter | `atomicAdd` on `bloom->misses` |
| Timing | `gettimeofday` | CUDA events (`cudaEventRecord`) |

Each kernel launches `ceil(n / blockSize)` blocks of `blockSize` threads. Thread `i` handles string `i` and threads past the end exit early. Concurrent inserts can write the same slot at once without atomics, because every writer stores the same value (`1`).

The GPU time covers the two kernel launches only. Memory allocation and host↔device copies are outside the timed region.

## Requirements

- An NVIDIA GPU and the CUDA Toolkit (`nvcc`)
- A C/C++ host compiler supported by your CUDA version

## Build and run

Compile with `nvcc`:

```bash
nvcc -O2 -o bloomfilter proj3-arefkhondaker.cu -lm
```

Run with three arguments:

```bash
./bloomfilter <num_elements> <false_positive_rate> <block_size>
```

| Argument | Meaning | Rules |
|---|---|---|
| `num_elements` | Number of random strings to generate, insert and query | Positive integer |
| `false_positive_rate` | Target false-positive probability `p` | Strictly between 0 and 1 |
| `block_size` | CUDA threads per block | Multiple of 32, at most 1024 |

Example:

```bash
./bloomfilter 10000 0.5 256
```

On the course server, use the provided run script instead:

```bash
/apps/GPU_course/runScript.sh proj3-arefkhondaker.cu 10000 0.5 256
```

## Output

The program prints one line of timing and one line of false negatives for each version:

```
[CPU] Insert+Query(or Total time of generation): <time> ms
[CPU] False negatives: <misses>/<num_elements>
[GPU] Insert+Query(or Total time of generation): <time> ms (<speedup>x speedup)
[GPU] False negatives: <misses>/<num_elements>
```

Speedup is CPU time divided by GPU time. Both false-negative counts should be `0`.

## Code map

| Function | Role |
|---|---|
| `siphash` / `device_siphash` | SipHash-2-4 on host / device |
| `init_filter` | Computes `m` and `k` from `n` and `p` |
| `add_to_filter` / `device_add_to_filter` | Inserts one string |
| `check_filter` / `device_check_filter` | Queries one string (0 = definitely absent, 1 = probably present) |
| `generate_flattened_string` | Builds the packed test strings and offset array |
| `addKernel` | Parallel insert, one thread per string |
| `checkKernel` | Parallel query, one thread per string, counts misses atomically |
| `main` | Parses arguments, runs and times both versions, prints results |

## Credits

- SipHash reference implementation © 2012–2022 Jean-Philippe Aumasson and Daniel J. Bernstein, dedicated to the public domain under CC0.
- Author: Aref Khondaker
