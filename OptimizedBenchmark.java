import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class OptimizedBenchmark {

    private static final int[] MATRIX_SIZES = {64, 128, 256, 512, 1024, 2048};
    private static final int[] SPARSE_LEVELS = {0, 50, 75, 90, 95};
    private static final int WARMUP_ITERATIONS = 5;
    private static final int REPETITIONS = 10;
    private static final String MATRIX_DIRECTORY = "./matrices";
    private static final String OUTPUT_CSV = "benchmark_raw_results.csv";

    private static final int BASE_CASE = 64;
    private static final int BLOCK = 16;

    private static final Map<Integer, Deque<int[]>> tempPools = new HashMap<>();

    public static void main(String[] args) {
        writeCsvHeader(OUTPUT_CSV);

        for (int size : MATRIX_SIZES) {
            for (int sparse : SPARSE_LEVELS) {
                System.out.println("\n============================================================");
                System.out.printf("[INFO] Starting configuration: Matrix Size = %d, Sparse = %d%%%n", size, sparse);
                System.out.println("============================================================\n");

                int totalIterations = WARMUP_ITERATIONS + REPETITIONS;

                for (int iter = 1; iter <= totalIterations; iter++) {
                    int[] A = loadMatrixFromBinFlat("A", size, sparse, MATRIX_DIRECTORY);
                    int[] B = loadMatrixFromBinFlat("B", size, sparse, MATRIX_DIRECTORY);

                    if (A == null || B == null) {
                        System.out.printf("Skipping size=%d, sparse=%d%% (missing matrices).%n", size, sparse);
                        continue;
                    }

                    String runId = "run_" + UUID.randomUUID().toString().substring(0, 8);
                    String timestamp = DateTimeFormatter.ISO_LOCAL_DATE_TIME.format(LocalDateTime.now());
                    String notes = "";
                    String warmup = (iter <= WARMUP_ITERATIONS) ? "1" : "0";

                    try {
                        Result result = strassenMatrixMultiplicationBenchmark(A, B, size);
                        double execTimeMs = result.executionTimeSec * 1000.0;
                        if (notes.trim().isEmpty()) notes = "No notes";

                        System.out.printf("[%s] size=%d, sparse=%d%% | optimized (Strassen) | time=%.2f ms | mem=%.2f MB | warm-up=%s | notes: %s%n",
                                runId, size, sparse, execTimeMs, result.memoryUsageMB, warmup, notes);

                        appendResultToCsv(OUTPUT_CSV, Arrays.asList(
                                runId, String.valueOf(size), String.valueOf(sparse), "optimized-strassen",
                                String.format("%.3f", execTimeMs), String.format("%.3f", result.memoryUsageMB),
                                String.valueOf(iter), timestamp, warmup, notes
                        ));
                    } catch (Exception e) {
                        notes = "Error: " + e.getMessage();
                        System.out.printf("[ERROR] %s failed for size=%d, sparse=%d%% - %s%n", runId, size, sparse, e);

                        appendResultToCsv(OUTPUT_CSV, Arrays.asList(
                                runId, String.valueOf(size), String.valueOf(sparse), "optimized-strassen",
                                "", "", String.valueOf(iter), timestamp, warmup, notes
                        ));
                    } finally {
                        tempPools.clear();
                    }
                }
            }
        }

        System.out.printf("%n[OK] All runs completed. Results saved to: %s%n", OUTPUT_CSV);
    }

    static class Result {
        double executionTimeSec;
        double memoryUsageMB;
        Result(double time, double mem) {
            this.executionTimeSec = time;
            this.memoryUsageMB = mem;
        }
    }

    private static int[] loadMatrixFromBinFlat(String label, int size, int sparse, String directory) {
        String filename = String.format("%s_%d_%d.bin", label, size, sparse);
        Path filepath = Paths.get(directory, filename);
        if (!Files.exists(filepath)) return null;

        int[] flat = new int[size * size];
        try (FileChannel fc = FileChannel.open(filepath, StandardOpenOption.READ)) {
            ByteBuffer buffer = ByteBuffer.allocate(size * size * Integer.BYTES);
            fc.read(buffer);
            buffer.flip();
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            buffer.asIntBuffer().get(flat);
        } catch (IOException e) {
            System.out.println("[ERROR] Error reading matrix: " + e.getMessage());
            return null;
        }
        return flat;
    }

    private static Result strassenMatrixMultiplicationBenchmark(int[] A, int[] B, int n) {
        Runtime runtime = Runtime.getRuntime();
        runtime.gc();
        long startMem = runtime.totalMemory() - runtime.freeMemory();
        long startTime = System.nanoTime();

        int[] C = strassenMatrixMultiplication(A, B, n);

        long endTime = System.nanoTime();
        long endMem = runtime.totalMemory() - runtime.freeMemory();
        double peakMemMB = (Math.max(endMem, startMem) - startMem) / (1024.0 * 1024.0);
        double elapsedSec = (endTime - startTime) / 1e9;

        return new Result(elapsedSec, peakMemMB);
    }

    private static int[] strassenMatrixMultiplication(int[] A, int[] B, int n) {
        int[] C = new int[n * n];
        tempPools.clear();
        strassenRecursive(A, 0, B, 0, C, 0, n, n);
        return C;
    }

    private static void strassenRecursive(int[] A, int aOff, int[] B, int bOff, int[] C, int cOff, int n, int stride) {
        if (n <= BASE_CASE) {
            multiplyBase(A, aOff, B, bOff, C, cOff, n, stride);
            return;
        }

        int newSize = n / 2;

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

        int[] S1 = getTemp(newSize);
        int[] S2 = getTemp(newSize);
        int[] M1 = getTemp(newSize);
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

        // --- Use helper methods with stride ---
        addInto(S1, 0, A, a11, stride, A, a22, stride, newSize);
        addInto(S2, 0, B, b11, stride, B, b22, stride, newSize);
        strassenRecursive(S1, 0, S2, 0, M1, 0, newSize, newSize);

        addInto(S3, 0, A, a21, stride, A, a22, stride, newSize);
        strassenRecursive(S3, 0, B, b11, M2, 0, newSize, newSize);

        subInto(S4, 0, B, b12, stride, B, b22, stride, newSize);
        strassenRecursive(A, a11, B, 0, M3, 0, newSize, newSize);

        subInto(S5, 0, B, b21, stride, B, b11, stride, newSize);
        strassenRecursive(A, a22, S5, 0, M4, 0, newSize, newSize);

        addInto(S6, 0, A, a11, stride, A, a12, stride, newSize);
        strassenRecursive(S6, 0, B, b22, M5, 0, newSize, newSize);

        subInto(S7, 0, A, a21, stride, A, a11, stride, newSize);
        addInto(S8, 0, B, b11, stride, B, b12, stride, newSize);
        strassenRecursive(S7, 0, S8, 0, M6, 0, newSize, newSize);

        subInto(S1, 0, A, a12, stride, A, a22, stride, newSize);
        addInto(S2, 0, B, b21, stride, B, b22, stride, newSize);
        strassenRecursive(S1, 0, S2, 0, M7, 0, newSize, newSize);

        combineC(C, c11, M1, M4, M5, M7, newSize, stride);
        addToC(C, c12, M3, M5, newSize, stride);
        addToC(C, c21, M2, M4, newSize, stride);
        combineC22(C, c22, M1, M2, M3, M6, newSize, stride);

        releaseTemp(S1); releaseTemp(S2); releaseTemp(M1);
        releaseTemp(S3); releaseTemp(M2); releaseTemp(S4);
        releaseTemp(M3); releaseTemp(S5); releaseTemp(M4);
        releaseTemp(S6); releaseTemp(M5); releaseTemp(S7);
        releaseTemp(M6); releaseTemp(S8); releaseTemp(M7);
    }

    private static void multiplyBase(int[] A, int aOff, int[] B, int bOff, int[] C, int cOff, int n, int stride) {
        for (int i = 0; i < n; i++) Arrays.fill(C, cOff + i * stride, cOff + i * stride + n, 0);
        for (int ii = 0; ii < n; ii += BLOCK) {
            int iimax = Math.min(ii + BLOCK, n);
            for (int kk = 0; kk < n; kk += BLOCK) {
                int kkmax = Math.min(kk + BLOCK, n);
                for (int i = ii; i < iimax; i++) {
                    int aRow = aOff + i * stride;
                    int cRow = cOff + i * stride;
                    for (int k = kk; k < kkmax; k++) {
                        int bRow = bOff + k * stride;
                        int aVal = A[aRow + k];
                        for (int j = 0; j < n; j++) C[cRow + j] += aVal * B[bRow + j];
                    }
                }
            }
        }
    }

    private static void addInto(int[] dest, int dOff, int[] A, int aOff, int aStride, int[] B, int bOff, int bStride, int n) {
        for (int i = 0; i < n; i++) {
            int dRow = dOff + i * n;
            int aRow = aOff + i * aStride;
            int bRow = bOff + i * bStride;
            for (int j = 0; j < n; j++) dest[dRow + j] = A[aRow + j] + B[bRow + j];
        }
    }

    private static void subInto(int[] dest, int dOff, int[] A, int aOff, int aStride, int[] B, int bOff, int bStride, int n) {
        for (int i = 0; i < n; i++) {
            int dRow = dOff + i * n;
            int aRow = aOff + i * aStride;
            int bRow = bOff + i * bStride;
            for (int j = 0; j < n; j++) dest[dRow + j] = A[aRow + j] - B[bRow + j];
        }
    }

    private static void addToC(int[] C, int cOff, int[] X, int[] Y, int n, int stride) {
        for (int i = 0; i < n; i++) {
            int cRow = cOff + i * stride;
            for (int j = 0; j < n; j++) C[cRow + j] = X[i * n + j] + Y[i * n + j];
        }
    }

    private static void combineC(int[] C, int cOff, int[] M1, int[] M4, int[] M5, int[] M7, int n, int stride) {
        for (int i = 0; i < n; i++) {
            int cRow = cOff + i * stride;
            for (int j = 0; j < n; j++) C[cRow + j] = M1[i * n + j] + M4[i * n + j] - M5[i * n + j] + M7[i * n + j];
        }
    }

    private static void combineC22(int[] C, int cOff, int[] M1, int[] M2, int[] M3, int[] M6, int n, int stride) {
        for (int i = 0; i < n; i++) {
            int cRow = cOff + i * stride;
            for (int j = 0; j < n; j++) C[cRow + j] = M1[i * n + j] - M2[i * n + j] + M3[i * n + j] + M6[i * n + j];
        }
    }

    private static int[] getTemp(int n) {
        int size = n * n;
        Deque<int[]> deque = tempPools.computeIfAbsent(n, k -> new ArrayDeque<>());
        int[] arr = deque.pollFirst();
        if (arr == null || arr.length < size) arr = new int[size];
        else Arrays.fill(arr, 0);
        return arr;
    }

    private static void releaseTemp(int[] arr) {
        if (arr == null) return;
        int n = (int) Math.round(Math.sqrt(arr.length));
        tempPools.computeIfAbsent(n, k -> new ArrayDeque<>()).offerFirst(arr);
    }

    private static void writeCsvHeader(String path) {
        File file = new File(path);
        if (!file.exists()) {
            try (PrintWriter writer = new PrintWriter(new FileWriter(file))) {
                writer.println(String.join(";", Arrays.asList(
                        "run_id","matrix_size","sparse_level_percent","implementation",
                        "execution_time_ms","memory_usage_mb","repetition","timestamp","warm-up","notes"
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
