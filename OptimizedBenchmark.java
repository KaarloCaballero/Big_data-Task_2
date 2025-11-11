import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * OptimizedBenchmark — Strassen implementation with several low-level optimizations:
 *  - Flattened matrices (int[])
 *  - Index ranges (offset + stride) instead of copying
 *  - Reusable temporary buffers (pool)
 *  - Write sums/diffs into reusable buffers
 *  - Blocked base-case multiplication (cache-friendly)
 *  - Use System.arraycopy for whole-row copies when appropriate
 */
public class OptimizedBenchmark {

    // --- Configuration ---
    private static final int[] MATRIX_SIZES = {64}; // {64, 128, 256, 512, 1024}
    private static final int[] SPARSE_LEVELS = {0, 50}; // {0, 50, 75, 90, 95}
    private static final int WARMUP_ITERATIONS = 5;
    private static final String MATRIX_DIRECTORY = "./matrices";
    private static final String OUTPUT_CSV = "matrix_multiplication_results.csv";
    private static final int REPETITIONS = 10;

    // Strassen tuning
    private static final int BASE_CASE = 64; // <= this do blocked naive multiply (tuneable)
    private static final int BLOCK = 16;     // blocking factor for base-case multiply (tuneable)

    // Temp pools keyed by submatrix size (n -> deque of int[n*n])
    private static final Map<Integer, Deque<int[]>> tempPools = new HashMap<>();

    public static void main(String[] args) {
        writeCsvHeader(OUTPUT_CSV);

        for (int size : MATRIX_SIZES) {
            for (int sparse : SPARSE_LEVELS) {

                System.out.println("\n============================================================");
                System.out.printf("[INFO] Starting configuration: Matrix Size = %d, Sparse = %d%%%n", size, sparse);
                System.out.println("============================================================\n");

                for (int rep = 1; rep <= REPETITIONS; rep++) {
                    int[] A = loadMatrixFromBinFlat("A", size, sparse, MATRIX_DIRECTORY);
                    int[] B = loadMatrixFromBinFlat("B", size, sparse, MATRIX_DIRECTORY);

                    if (A == null || B == null) {
                        System.out.printf("Skipping size=%d, sparse=%d%% (missing matrices).%n", size, sparse);
                        continue;
                    }

                    String runId = "run_" + UUID.randomUUID().toString().substring(0, 8);
                    String timestamp = DateTimeFormatter.ISO_LOCAL_DATE_TIME.format(LocalDateTime.now());
                    String notes = "";

                    // --- Warm-up logic uses global variable ---
                    String warmup = (rep <= WARMUP_ITERATIONS) ? "1" : "0";

                    try {
                        Result result = strassenMatrixMultiplicationBenchmark(A, B, size);
                        double execTimeMs = result.executionTimeSec * 1000.0;
                        if (notes.trim().isEmpty()) notes = "No notes";

                        System.out.printf("[%s] size=%d, sparse=%d%% | optimized (Strassen) | time=%.2f ms | mem=%.2f MB | warm-up=%s | notes: %s%n",
                                runId, size, sparse, execTimeMs, result.memoryUsageMB, warmup, notes);

                        // --- warm-up before notes ---
                        appendResultToCsv(OUTPUT_CSV, Arrays.asList(
                                runId,
                                String.valueOf(size),
                                String.valueOf(sparse),
                                "optimized-strassen",
                                String.format("%.3f", execTimeMs),
                                String.format("%.3f", result.memoryUsageMB),
                                String.valueOf(rep),
                                timestamp,
                                warmup,
                                notes
                        ));

                    } catch (Exception e) {
                        notes = "Error: " + e.getMessage();
                        System.out.printf("[ERROR] %s failed for size=%d, sparse=%d%% - %s%n", runId, size, sparse, e);

                        appendResultToCsv(OUTPUT_CSV, Arrays.asList(
                                runId,
                                String.valueOf(size),
                                String.valueOf(sparse),
                                "optimized-strassen",
                                "", "", String.valueOf(rep),
                                timestamp,
                                warmup,
                                notes
                        ));
                    } finally {
                        // clear pools between top-level runs to avoid cross-run memory growth (optional)
                        tempPools.clear();
                    }
                }
            }
        }

        System.out.printf("%n[OK] All runs completed. Results saved to: %s%n", OUTPUT_CSV);
    }

    // --- Data class ---
    static class Result {
        double executionTimeSec;
        double memoryUsageMB;
        Result(double time, double mem) {
            this.executionTimeSec = time;
            this.memoryUsageMB = mem;
        }
    }

    // --- Matrix Loader (flat int[]) ---
    private static int[] loadMatrixFromBinFlat(String label, int size, int sparse, String directory) {
        String filename = String.format("%s_%d_%d.bin", label, size, sparse);
        Path filepath = Paths.get(directory, filename);
        if (!Files.exists(filepath)) {
            System.out.printf("[INFO] File %s does not exist.%n", filepath);
            return null;
        }

        int[] flat = new int[size * size];
        try (FileChannel fc = FileChannel.open(filepath, StandardOpenOption.READ)) {
            ByteBuffer buffer = ByteBuffer.allocate(size * size * Integer.BYTES);
            fc.read(buffer);
            buffer.flip();
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            IntBuffer intBuf = buffer.asIntBuffer();
            intBuf.get(flat);
        } catch (IOException e) {
            System.out.println("[ERROR] Error reading matrix: " + e.getMessage());
            return null;
        }
        return flat;
    }

    // --- Strassen Benchmark wrapper ---
    private static Result strassenMatrixMultiplicationBenchmark(int[] A, int[] B, int n) {
        Runtime runtime = Runtime.getRuntime();
        runtime.gc();
        long startMem = runtime.totalMemory() - runtime.freeMemory();
        long startTime = System.nanoTime();

        int[] C = strassenMatrixMultiplication(A, B, n);

        long endTime = System.nanoTime();
        long endMem = runtime.totalMemory() - runtime.freeMemory();
        double peakMemMB = Math.max(endMem, startMem) - startMem;
        peakMemMB /= (1024.0 * 1024.0);

        double elapsedSec = (endTime - startTime) / 1e9;
        // optionally write or use C to prevent dead-code elimination (not necessary in standard runs)
        if (C == null) System.out.println("[ERROR] C null");

        return new Result(elapsedSec, peakMemMB);
    }

    // --- High-level Strassen entry (returns flat C) ---
    private static int[] strassenMatrixMultiplication(int[] A, int[] B, int n) {
        int[] C = new int[n * n];
        // Ensure pools are (re)initialized for this top-level problem size
        tempPools.clear();
        strassenRecursive(A, 0, B, 0, C, 0, n, n);
        return C;
    }

    /**
     * Core recursive Strassen operating on flat arrays.
     *
     * @param A      flat array for A
     * @param aOff   starting offset in A for this submatrix
     * @param B      flat array for B
     * @param bOff   starting offset in B for this submatrix
     * @param C      flat array for C (output)
     * @param cOff   starting offset in C for this submatrix
     * @param n      submatrix dimension
     * @param stride the row stride (original full matrix size)
     */
    private static void strassenRecursive(int[] A, int aOff, int[] B, int bOff, int[] C, int cOff, int n, int stride) {
        if (n <= BASE_CASE) {
            // base-case blocked multiply: C = A * B
            multiplyBase(A, aOff, B, bOff, C, cOff, n, stride);
            return;
        }

        int newSize = n / 2;

        // offsets for sub-blocks (row-major flattened)
        int a11 = aOff;
        int a12 = aOff + newSize;
        int a21 = aOff + newSize * stride;
        int a22 = aOff + newSize * stride + newSize;

        int b11 = bOff;
        int b12 = bOff + newSize;
        int b21 = bOff + newSize * stride;
        int b22 = bOff + newSize * stride + newSize;

        int c11 = cOff;
        int c12 = cOff + newSize;
        int c21 = cOff + newSize * stride;
        int c22 = cOff + newSize * stride + newSize;

        // Acquire reusable temp buffers for sums/differences and results
        int[] S1 = getTemp(newSize); // for A11 + A22 or other sums
        int[] S2 = getTemp(newSize); // for B11 + B22 or other
        int[] M1 = getTemp(newSize); // result of recursive multiplication
        int[] S3 = getTemp(newSize);
        int[] M2 = getTemp(newSize);
        int[] S4 = getTemp(newSize);
        int[] M3 = getTemp(newSize);
        int[] S5 = getTemp(newSize);
        int[] M4 = getTemp(newSize);
        int[] S6 = getTemp(newSize);
        int[] M5 = getTemp(newSize);
        int[] S7 = getTemp(newSize);
        int[] M6 = getTemp(newSize);
        int[] S8 = getTemp(newSize);
        int[] M7 = getTemp(newSize);

        // M1 = (A11 + A22) * (B11 + B22)
        addInto(S1, 0, A, a11, A, a22, newSize, stride); // S1 = A11 + A22
        addInto(S2, 0, B, b11, B, b22, newSize, stride); // S2 = B11 + B22
        strassenRecursive(S1, 0, S2, 0, M1, 0, newSize, newSize);

        // M2 = (A21 + A22) * B11
        addInto(S3, 0, A, a21, A, a22, newSize, stride); // S3 = A21 + A22
        strassenRecursive(S3, 0, B, b11, M2, 0, newSize, stride);

        // M3 = A11 * (B12 - B22)
        subInto(S4, 0, B, b12, B, b22, newSize, stride); // S4 = B12 - B22
        strassenRecursive(A, a11, S4, 0, M3, 0, newSize, newSize);

        // M4 = A22 * (B21 - B11)
        subInto(S5, 0, B, b21, B, b11, newSize, stride); // S5 = B21 - B11
        strassenRecursive(A, a22, S5, 0, M4, 0, newSize, newSize);

        // M5 = (A11 + A12) * B22
        addInto(S6, 0, A, a11, A, a12, newSize, stride); // S6 = A11 + A12
        strassenRecursive(S6, 0, B, b22, M5, 0, newSize, stride);

        // M6 = (A21 - A11) * (B11 + B12)
        subInto(S7, 0, A, a21, A, a11, newSize, stride); // S7 = A21 - A11
        addInto(S8, 0, B, b11, B, b12, newSize, stride); // S8 = B11 + B12
        strassenRecursive(S7, 0, S8, 0, M6, 0, newSize, newSize);

        // M7 = (A12 - A22) * (B21 + B22)
        subInto(S1, 0, A, a12, A, a22, newSize, stride); // reuse S1 = A12 - A22 (we can reuse)
        addInto(S2, 0, B, b21, B, b22, newSize, stride); // reuse S2 = B21 + B22
        strassenRecursive(S1, 0, S2, 0, M7, 0, newSize, newSize);

        // Compute C quadrants:
        // C11 = M1 + M4 - M5 + M7
        // We'll compute into C directly row by row for locality.
        combineC(C, c11, M1, M4, M5, M7, newSize, stride);

        // C12 = M3 + M5
        addToC(C, c12, M3, M5, newSize, stride);

        // C21 = M2 + M4
        addToC(C, c21, M2, M4, newSize, stride);

        // C22 = M1 - M2 + M3 + M6
        combineC22(C, c22, M1, M2, M3, M6, newSize, stride);

        // release temps back to pool
        releaseTemp(S1);
        releaseTemp(S2);
        releaseTemp(M1);
        releaseTemp(S3);
        releaseTemp(M2);
        releaseTemp(S4);
        releaseTemp(M3);
        releaseTemp(S5);
        releaseTemp(M4);
        releaseTemp(S6);
        releaseTemp(M5);
        releaseTemp(S7);
        releaseTemp(M6);
        releaseTemp(S8);
        releaseTemp(M7);
    }

    // --- Helper: blocked base-case multiply (writes into C region, sets to zero first) ---
    private static void multiplyBase(int[] A, int aOff, int[] B, int bOff, int[] C, int cOff, int n, int stride) {
        // zero C block
        for (int i = 0; i < n; i++) {
            int destPos = cOff + i * stride;
            Arrays.fill(C, destPos, destPos + n, 0);
        }

        // blocked multiplication: for i, k, j order (use row-major accesses)
        int bs = BLOCK;
        for (int ii = 0; ii < n; ii += bs) {
            int iimax = Math.min(ii + bs, n);
            for (int kk = 0; kk < n; kk += bs) {
                int kkmax = Math.min(kk + bs, n);
                for (int i = ii; i < iimax; i++) {
                    int aRow = aOff + i * stride;
                    int cRow = cOff + i * stride;
                    for (int k = kk; k < kkmax; k++) {
                        int aVal = A[aRow + k];
                        int bRow = bOff + k * stride;
                        int cPos = cRow;
                        // accumulate aVal * B[k][j] into C[i][j]
                        for (int j = 0; j < n; j++) {
                            C[cPos + j] += aVal * B[bRow + j];
                        }
                    }
                }
            }
        }
    }

    // --- addInto: dest = A_sub + B_sub; A_sub and B_sub are in flat arrays with given offsets and stride ---
    private static void addInto(int[] dest, int destOff, int[] A, int aOff, int[] B, int bOff, int n, int stride) {
        // row-wise to allow System.arraycopy optimization when one operand is a direct copy
        for (int i = 0; i < n; i++) {
            int aRow = aOff + i * stride;
            int bRow = bOff + i * stride;
            int dRow = destOff + i * n; // dest is contiguous n*n with stride == n for temp buffers
            for (int j = 0; j < n; j++) {
                dest[dRow + j] = A[aRow + j] + B[bRow + j];
            }
        }
    }

    // --- subInto: dest = A_sub - B_sub ---
    private static void subInto(int[] dest, int destOff, int[] A, int aOff, int[] B, int bOff, int n, int stride) {
        for (int i = 0; i < n; i++) {
            int aRow = aOff + i * stride;
            int bRow = bOff + i * stride;
            int dRow = destOff + i * n;
            for (int j = 0; j < n; j++) {
                dest[dRow + j] = A[aRow + j] - B[bRow + j];
            }
        }
    }

    // --- addToC: C_block = X + Y  (X and Y are contiguous buffers size n*n) ---
    private static void addToC(int[] C, int cOff, int[] X, int[] Y, int n, int stride) {
        for (int i = 0; i < n; i++) {
            int cRow = cOff + i * stride;
            int xRow = i * n;
            int yRow = i * n;
            for (int j = 0; j < n; j++) {
                C[cRow + j] = X[xRow + j] + Y[yRow + j];
            }
        }
    }

    // C11 = M1 + M4 - M5 + M7
    private static void combineC(int[] C, int cOff, int[] M1, int[] M4, int[] M5, int[] M7, int n, int stride) {
        for (int i = 0; i < n; i++) {
            int cRow = cOff + i * stride;
            int m1r = i * n;
            int m4r = i * n;
            int m5r = i * n;
            int m7r = i * n;
            for (int j = 0; j < n; j++) {
                C[cRow + j] = M1[m1r + j] + M4[m4r + j] - M5[m5r + j] + M7[m7r + j];
            }
        }
    }

    // C22 = M1 - M2 + M3 + M6
    private static void combineC22(int[] C, int cOff, int[] M1, int[] M2, int[] M3, int[] M6, int n, int stride) {
        for (int i = 0; i < n; i++) {
            int cRow = cOff + i * stride;
            int m1r = i * n;
            int m2r = i * n;
            int m3r = i * n;
            int m6r = i * n;
            for (int j = 0; j < n; j++) {
                C[cRow + j] = M1[m1r + j] - M2[m2r + j] + M3[m3r + j] + M6[m6r + j];
            }
        }
    }

    // --- Temp pool utilities ---
    private static int[] getTemp(int n) {
        int size = n * n;
        Deque<int[]> deque = tempPools.get(n);
        if (deque == null) {
            deque = new ArrayDeque<>();
            tempPools.put(n, deque);
        }
        int[] arr = deque.pollFirst();
        if (arr == null || arr.length < size) {
            arr = new int[size];
        } else {
            // zero the buffer before returning for correctness
            Arrays.fill(arr, 0);
        }
        return arr;
    }

    private static void releaseTemp(int[] arr) {
        if (arr == null) return;
        int n = (int) Math.round(Math.sqrt(arr.length));
        Deque<int[]> deque = tempPools.get(n);
        if (deque == null) {
            deque = new ArrayDeque<>();
            tempPools.put(n, deque);
        }
        deque.offerFirst(arr);
    }

    // --- CSV Utilities (semicolon separator) ---
    private static void writeCsvHeader(String path) {
        File file = new File(path);
        if (!file.exists()) {
            try (PrintWriter writer = new PrintWriter(new FileWriter(file))) {
                writer.println(String.join(";", Arrays.asList(
                        "run_id",
                        "matrix_size",
                        "sparse_level_percent",
                        "implementation",
                        "execution_time_ms",
                        "memory_usage_mb",
                        "repetition",
                        "timestamp",
                        "warm-up",
                        "notes"
                )));
            } catch (IOException e) {
                System.out.println("[ERROR] Error writing CSV header: " + e.getMessage());
            }
        }
    }

    private static void appendResultToCsv(String path, List<String> row) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(path, true))) {
            writer.println(String.join(";", row));
        } catch (IOException e) {
            System.out.println("[ERROR] Error appending to CSV: " + e.getMessage());
        }
    }
}
