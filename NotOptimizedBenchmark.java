import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import java.time.*;
import java.time.format.*;
import java.util.UUID;
import java.nio.channels.FileChannel;

public class NotOptimizedBenchmark {

    // --- Configuration ---
    private static final int[] MATRIX_SIZES = {64, 128, 256, 512, 1024, 2048};
    private static final int[] SPARSE_LEVELS = {0, 50, 75, 90, 95};
    private static final int WARMUP_ITERATIONS = 5;
    private static final String MATRIX_DIRECTORY = "./matrices";
    private static final String OUTPUT_CSV = "matrix_multiplication_results.csv";
    private static final int REPETITIONS = 10;

    public static void main(String[] args) {
        writeCsvHeader(OUTPUT_CSV);

        for (int size : MATRIX_SIZES) {
            for (int sparse : SPARSE_LEVELS) {

                System.out.println("\n============================================================");
                System.out.printf("[INFO] Starting configuration: Matrix Size = %d, Sparse = %d%%%n", size, sparse);
                System.out.println("============================================================\n");

                int totalIterations = WARMUP_ITERATIONS + REPETITIONS;

                for (int rep = 1; rep <= totalIterations; rep++) {
                    int[][] A = loadMatrixFromBin("A", size, sparse, MATRIX_DIRECTORY);
                    int[][] B = loadMatrixFromBin("B", size, sparse, MATRIX_DIRECTORY);

                    if (A == null || B == null) {
                        System.out.printf("Skipping size=%d, sparse=%d%% (missing matrices).%n", size, sparse);
                        continue;
                    }

                    String runId = "run_" + UUID.randomUUID().toString().substring(0, 8);
                    String timestamp = DateTimeFormatter.ISO_LOCAL_DATE_TIME.format(LocalDateTime.now());
                    String notes = "";

                    // Determine if this iteration is a warm-up
                    int warmUp = (rep <= WARMUP_ITERATIONS) ? 1 : 0;

                    try {
                        Result result = naiveMatrixMultiplication(A, B, size);
                        double execTimeMs = result.executionTimeSec * 1000.0;

                        if (notes.trim().isEmpty()) {
                            notes = "No notes";
                        }

                        System.out.printf("[%s] size=%d, sparse=%d%% | unoptimized | time=%.2f ms | mem=%.2f MB | warm-up=%d | notes: %s%n",
                                runId, size, sparse, execTimeMs, result.memoryUsageMB, warmUp, notes);

                        appendResultToCsv(OUTPUT_CSV, Arrays.asList(
                                runId,
                                String.valueOf(size),
                                String.valueOf(sparse),
                                "unoptimized",
                                String.format("%.3f", execTimeMs),
                                String.format("%.3f", result.memoryUsageMB),
                                String.valueOf(rep),
                                timestamp,
                                String.valueOf(warmUp),
                                notes
                        ));

                    } catch (Exception e) {
                        notes = "Error: " + e.getMessage();
                        System.out.printf("[ERROR] %s failed for size=%d, sparse=%d%% - %s%n", runId, size, sparse, e);
                        appendResultToCsv(OUTPUT_CSV, Arrays.asList(
                                runId, String.valueOf(size), String.valueOf(sparse),
                                "unoptimized", "", "", String.valueOf(rep), timestamp,
                                String.valueOf(warmUp), notes
                        ));
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

    // --- Matrix Loader ---
    private static int[][] loadMatrixFromBin(String label, int size, int sparse, String directory) {
        String filename = String.format("%s_%d_%d.bin", label, size, sparse);
        Path filepath = Paths.get(directory, filename);
        if (!Files.exists(filepath)) {
            System.out.printf("[INFO] File %s does not exist.%n", filepath);
            return null;
        }

        int[][] matrix = new int[size][size];
        try (FileChannel fc = FileChannel.open(filepath, StandardOpenOption.READ)) {
            ByteBuffer buffer = ByteBuffer.allocate(size * size * Integer.BYTES);
            fc.read(buffer);
            buffer.flip();
            buffer.order(ByteOrder.LITTLE_ENDIAN);

            IntBuffer intBuf = buffer.asIntBuffer();
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    matrix[i][j] = intBuf.get();
                }
            }
        } catch (IOException e) {
            System.out.println("[ERROR] Error reading matrix: " + e.getMessage());
            return null;
        }
        return matrix;
    }

    // --- Naive Matrix Multiplication ---
    private static Result naiveMatrixMultiplication(int[][] A, int[][] B, int n) {
        Runtime runtime = Runtime.getRuntime();
        runtime.gc();
        long startMem = (runtime.totalMemory() - runtime.freeMemory());
        long startTime = System.nanoTime();

        int[][] C = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }

        long endTime = System.nanoTime();
        long endMem = (runtime.totalMemory() - runtime.freeMemory());
        double peakMemMB = Math.max(endMem, startMem) - startMem;
        peakMemMB /= (1024.0 * 1024.0);

        double elapsedSec = (endTime - startTime) / 1e9;

        return new Result(elapsedSec, peakMemMB);
    }

    // --- CSV Utilities ---
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
