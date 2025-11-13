import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class NotOptimizedBenchmark {

    private static final int[] MATRIX_SIZES = {64, 128, 256, 512};
    private static final int[] SPARSE_LEVELS = {0, 50, 75, 90, 95};
    private static final int WARMUP_ITERATIONS = 5;
    private static final int REPETITIONS = 10;
    private static final String MATRIX_DIRECTORY = "./matrices";
    private static final String OUTPUT_CSV = "benchmark_raw_results.csv";

    public static void main(String[] args) {
        writeCsvHeader(OUTPUT_CSV);

        for (int size : MATRIX_SIZES) {
            for (int sparse : SPARSE_LEVELS) {
                System.out.printf("[INFO] Starting size=%d, sparse=%d%%%n", size, sparse);

                int totalIterations = WARMUP_ITERATIONS + REPETITIONS;

                for (int rep = 1; rep <= totalIterations; rep++) {
                    int[][] A = loadDenseMatrix("A", size, sparse, MATRIX_DIRECTORY);
                    int[][] B = loadDenseMatrix("B", size, sparse, MATRIX_DIRECTORY);

                    if (A == null || B == null) continue;

                    String runId = "run_" + UUID.randomUUID().toString().substring(0, 8);
                    String timestamp = DateTimeFormatter.ISO_LOCAL_DATE_TIME.format(LocalDateTime.now());
                    int warmUp = (rep <= WARMUP_ITERATIONS) ? 1 : 0;

                    try {
                        Result result = cooSparseMultiplicationBenchmark(A, B, size);

                        appendResultToCsv(OUTPUT_CSV, Arrays.asList(
                                runId, String.valueOf(size), String.valueOf(sparse),
                                "unoptimized",
                                String.format("%.3f", result.executionTimeSec * 1000),
                                String.format("%.3f", result.memoryUsageMB),
                                String.valueOf(rep), timestamp, String.valueOf(warmUp), "No notes"
                        ));

                        System.out.printf("[%s] size=%d, sparse=%d%% | time=%.2f ms | mem=%.2f MB%n",
                                runId, size, sparse, result.executionTimeSec * 1000, result.memoryUsageMB);

                    } catch (Exception e) {
                        System.out.printf("[ERROR] %s failed: %s%n", runId, e.getMessage());
                    }
                    try {
                        Thread.sleep(2000);
                    } catch (InterruptedException e){
                        System.out.print("Pause was interrupted, continuing...");
                    }
                }
            }
        }

        System.out.println("[OK] Benchmark completed.");
    }

    // --- Load dense matrix from file ---
    private static int[][] loadDenseMatrix(String label, int size, int sparse, String directory) {
        String filename = String.format("%s_%d_%d.bin", label, size, sparse);
        Path filepath = Paths.get(directory, filename);
        if (!Files.exists(filepath)) return null;

        int[][] dense = new int[size][size];
        try (FileChannel fc = FileChannel.open(filepath, StandardOpenOption.READ)) {
            ByteBuffer buffer = ByteBuffer.allocate(size * size * Integer.BYTES);
            fc.read(buffer);
            buffer.flip();
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            IntBuffer intBuf = buffer.asIntBuffer();
            for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++)
                    dense[i][j] = intBuf.get();
        } catch (IOException e) {
            System.out.println("[ERROR] Reading matrix: " + e.getMessage());
            return null;
        }

        return dense;
    }

    // --- COO-based multiplication benchmark ---
    private static Result cooSparseMultiplicationBenchmark(int[][] A, int[][] B, int n) {
        Runtime runtime = Runtime.getRuntime();
        runtime.gc();
        long startMem = runtime.totalMemory() - runtime.freeMemory();
        long startTime = System.nanoTime();

        int[][] C = multiplyCOO(A, B, n);

        long endTime = System.nanoTime();
        long endMem = runtime.totalMemory() - runtime.freeMemory();
        double peakMemMB = (Math.max(endMem, startMem) - startMem) / (1024.0 * 1024.0);
        double elapsedSec = (endTime - startTime) / 1e9;

        return new Result(elapsedSec, peakMemMB);
    }

    // --- Convert dense matrix to COO (Coordinate list) ---
    private static List<COOEntry> toCOO(int[][] M, int n) {
        List<COOEntry> entries = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int v = M[i][j];
                if (v != 0)
                    entries.add(new COOEntry(i, j, v));
            }
        }
        return entries;
    }

    // --- COO-based triple loop sparse multiplication ---
    private static int[][] multiplyCOO(int[][] A, int[][] B, int n) {
        List<COOEntry> aEntries = toCOO(A, n);
        List<COOEntry> bEntries = toCOO(B, n);
        int[][] C = new int[n][n];

        // Unoptimized COO triple loop
        for (COOEntry a : aEntries) {
            for (COOEntry b : bEntries) {
                if (a.col == b.row) {
                    C[a.row][b.col] += a.val * b.val;
                }
            }
        }

        return C;
    }

    // --- COO entry class ---
    private static class COOEntry {
        int row, col, val;
        COOEntry(int r, int c, int v) {
            this.row = r; this.col = c; this.val = v;
        }
    }

    // --- Result class ---
    static class Result {
        double executionTimeSec;
        double memoryUsageMB;
        Result(double time, double mem) {
            this.executionTimeSec = time;
            this.memoryUsageMB = mem;
        }
    }

    // --- CSV utilities ---
    private static void writeCsvHeader(String path) {
        File file = new File(path);
        if (!file.exists()) {
            try (PrintWriter writer = new PrintWriter(new FileWriter(file))) {
                writer.println(String.join(";", Arrays.asList(
                        "run_id","matrix_size","sparse_level_percent","implementation",
                        "execution_time_ms","memory_usage_mb","repetition","timestamp","warm-up","notes"
                )));
            } catch (IOException e) { System.out.println("[ERROR] CSV header: " + e.getMessage()); }
        }
    }

    private static void appendResultToCsv(String path, List<String> row) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(path, true))) {
            writer.println(String.join(";", row));
        } catch (IOException e) { System.out.println("[ERROR] CSV append: " + e.getMessage()); }
    }
}
