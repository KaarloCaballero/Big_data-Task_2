# ----------------------------
# PowerShell script to run JMH benchmarks with checks
# ----------------------------

function Check-ExitCode {
    param($stepDescription)
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] $stepDescription completed successfully." -ForegroundColor Green
    } else {
        Write-Host "[ERROR] $stepDescription failed with exit code $LASTEXITCODE." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# 1️⃣ Run generate_matrices.py
#Write-Host "`n============================================================"
#Write-Host "Step 1: Generating matrices..."
#Write-Host "============================================================`n"
#python .\generate_matrices.py
#Check-ExitCode "Matrix generation"

# 2️⃣ Compile NotOptimizedBenchmarkJMH.java
#Write-Host "`n============================================================"
#Write-Host "Step 2: Compiling NotOptimizedBenchmark.java..."
#Write-Host "============================================================`n"
#javac .\NotOptimizedBenchmark.java
#Check-ExitCode "NotOptimizedBenchmark compilation"

# 3️⃣ Compile OptimizedBenchmarkJMH.java
Write-Host "`n============================================================"
Write-Host "Step 3: Compiling OptimizedBenchmark.java..."
Write-Host "============================================================`n"
javac .\OptimizedBenchmark.java
Check-ExitCode "OptimizedBenchmark compilation"

# 4️⃣ Execute NotOptimizedBenchmarkJMH with JMH
#Write-Host "`n============================================================"
#Write-Host "Step 4: Running NotOptimizedBenchmark..."
#Write-Host "============================================================`n"
#java -XX:ActiveProcessorCount=16 NotOptimizedBenchmark
#Check-ExitCode "NotOptimizedBenchmark execution successfull"

# 5️⃣ Execute OptimizedBenchmarkJMH with JMH
Write-Host "`n============================================================"
Write-Host "Step 5: Running OptimizedBenchmark..."
Write-Host "============================================================`n"
java -Xmx4G -XX:+TieredCompilation -XX:ActiveProcessorCount=16 OptimizedBenchmark
Check-ExitCode "OptimizedBenchmark execution successfull"

Write-Host "`n[OK] All steps completed successfully." -ForegroundColor Green


