import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class OptimizedBenchmark {

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
                System.out.printf("[INFO] Matrix size=%d, sparse=%d%%%n", size, sparse);

                for (int iter = 1; iter <= WARMUP_ITERATIONS + REPETITIONS; iter++) {
                    SparseMatrix A = loadSparseMatrix("A", size, sparse, MATRIX_DIRECTORY);
                    SparseMatrix B = loadSparseMatrix("B", size, sparse, MATRIX_DIRECTORY);

                    if (A == null || B == null) continue;

                    String runId = "run_" + UUID.randomUUID().toString().substring(0, 8);
                    String timestamp = DateTimeFormatter.ISO_LOCAL_DATE_TIME.format(LocalDateTime.now());
                    boolean warmup = iter <= WARMUP_ITERATIONS;

                    long startMem = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
                    long startTime = System.nanoTime();

                    SparseMatrix C = gustavsonMultiply(A, B);

                    long endTime = System.nanoTime();
                    long endMem = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
                    double memMB = (endMem - startMem) / (1024.0 * 1024.0);
                    double timeMs = (endTime - startTime) / 1e6;

                    System.out.printf("[%s] size=%d, sparse=%d%% | time=%.2f ms | mem=%.2f MB | warmup=%b%n",
                            runId, size, sparse, timeMs, memMB, warmup);

                    appendResultToCsv(OUTPUT_CSV, Arrays.asList(
                            runId, String.valueOf(size), String.valueOf(sparse),
                            "optimized", String.format("%.3f", timeMs), String.format("%.3f", memMB),
                            String.valueOf(iter), timestamp, warmup ? "1" : "0", "No notes"
                    ));
                    try {
                        Thread.sleep(2000);
                    } catch (InterruptedException e){
                        System.out.print("Pause was interrupted, continuing...");
                    }
                }
            }
        }
    }

    // ===================== Sparse Matrix Representation (CSR) =====================
    static class SparseMatrix {
        int rows, cols;
        int[] rowPtr;
        int[] colIdx;
        double[] values;

        SparseMatrix(int rows, int cols, int nnz) {
            this.rows = rows;
            this.cols = cols;
            this.rowPtr = new int[rows + 1];
            this.colIdx = new int[nnz];
            this.values = new double[nnz];
        }
    }

    private static SparseMatrix loadSparseMatrix(String label, int size, int sparse, String dir) {
        String filename = String.format("%s_%d_%d.bin", label, size, sparse);
        Path path = Paths.get(dir, filename);
        if (!Files.exists(path)) return null;

        List<Integer> rowPtrList = new ArrayList<>();
        List<Integer> colIdxList = new ArrayList<>();
        List<Double> valList = new ArrayList<>();
        rowPtrList.add(0);

        try (FileChannel fc = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer buffer = ByteBuffer.allocate(size * size * Integer.BYTES);
            fc.read(buffer);
            buffer.flip();
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            IntBuffer intBuf = buffer.asIntBuffer();

            int nnz = 0;
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    int val = intBuf.get();
                    if (val != 0) {
                        colIdxList.add(j);
                        valList.add((double) val);
                        nnz++;
                    }
                }
                rowPtrList.add(nnz);
            }
        } catch (IOException e) {
            System.out.println("[ERROR] Failed to read matrix: " + e.getMessage());
            return null;
        }

        SparseMatrix mat = new SparseMatrix(size, size, valList.size());
        for (int i = 0; i < mat.rowPtr.length; i++) mat.rowPtr[i] = rowPtrList.get(i);
        for (int i = 0; i < mat.colIdx.length; i++) mat.colIdx[i] = colIdxList.get(i);
        for (int i = 0; i < mat.values.length; i++) mat.values[i] = valList.get(i);
        return mat;
    }

    // ===================== Gustavson's Algorithm for CSR × CSR =====================
    private static SparseMatrix gustavsonMultiply(SparseMatrix A, SparseMatrix B) {
        int n = A.rows;
        int[] rowPtrC = new int[n + 1];
        List<Integer> colIdxList = new ArrayList<>();
        List<Double> valList = new ArrayList<>();

        double[] temp = new double[n];
        int[] marker = new int[n]; // Tracks which columns have non-zero entries
        Arrays.fill(marker, -1);

        rowPtrC[0] = 0;

        for (int i = 0; i < n; i++) {
            int rowStart = A.rowPtr[i];
            int rowEnd = A.rowPtr[i + 1];
            int nnzRow = 0;

            for (int idxA = rowStart; idxA < rowEnd; idxA++) {
                int k = A.colIdx[idxA];
                double valA = A.values[idxA];

                int bStart = B.rowPtr[k];
                int bEnd = B.rowPtr[k + 1];

                for (int idxB = bStart; idxB < bEnd; idxB++) {
                    int j = B.colIdx[idxB];
                    double valB = B.values[idxB];

                    if (marker[j] != i) {
                        marker[j] = i;
                        temp[j] = valA * valB;
                        nnzRow++;
                    } else {
                        temp[j] += valA * valB;
                    }
                }
            }

            // Collect non-zero entries for this row
            for (int j = 0; j < n; j++) {
                if (marker[j] == i) {
                    colIdxList.add(j);
                    valList.add(temp[j]);
                    temp[j] = 0; // reset for next row
                }
            }
            rowPtrC[i + 1] = rowPtrC[i] + nnzRow;
        }

        SparseMatrix C = new SparseMatrix(n, n, valList.size());
        C.rowPtr = rowPtrC;
        for (int i = 0; i < valList.size(); i++) {
            C.colIdx[i] = colIdxList.get(i);
            C.values[i] = valList.get(i);
        }
        return C;
    }

    // ===================== CSV Utilities =====================
    private static void writeCsvHeader(String path) {
        File file = new File(path);
        if (!file.exists()) {
            try (PrintWriter writer = new PrintWriter(new FileWriter(file))) {
                writer.println(String.join(";", Arrays.asList(
                        "run_id", "matrix_size", "sparse_level_percent", "implementation",
                        "execution_time_ms", "memory_usage_mb", "repetition", "timestamp", "warm-up", "notes"
                )));
            } catch (IOException e) {
                System.out.println("[ERROR] CSV header write failed: " + e.getMessage());
            }
        }
    }

    private static void appendResultToCsv(String path, List<String> row) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(path, true))) {
            writer.println(String.join(";", row));
        } catch (IOException e) {
            System.out.println("[ERROR] CSV append failed: " + e.getMessage());
        }
    }
}
