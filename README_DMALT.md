## Naive kernel

```sh
Running kernel 21 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000243) s, performance: (   17.3) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.001868) s, performance: (   18.0) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.006513) s, performance: (   41.2) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.035448) s, performance: (   60.6) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.279186) s, performance: (   61.5) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
Average elapsed time: (2.239357) s, performance: (   61.4) GFLOPS. size: (4096).
==PROF== Connected to process 3520 (/content/sgemm/build/sgemm)
Running kernel 21 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000384) s, performance: (   10.9) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.001873) s, performance: (   17.9) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.006701) s, performance: (   40.1) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.035293) s, performance: (   60.8) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.278969) s, performance: (   61.6) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
==PROF== Profiling "sgemm_naive_dmalt": 0%..
==WARNING== Launching the workload is taking more time than expected. If this continues to hang, terminate the profile and re-try by profiling the range of all related launches using '--replay-mode range'. See https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#replay for more details.
..50%....100% - 8 passes
Average elapsed time: (3.242528) s, performance: (   42.4) GFLOPS. size: (4096).
==PROF== Disconnected from process 3520
[3520] sgemm@127.0.0.1
  sgemm_naive_dmalt(int, int, int, float, const float *, const float *, float, float *) (128, 128, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ----------------
    Metric Name             Metric Unit     Metric Value
    ----------------------- ----------- ----------------
    DRAM Frequency                  Ghz             5.00
    SM Frequency                    Mhz           585.00
    Elapsed Cycles                cycle    3,548,923,287
    Memory Throughput                 %            49.95
    DRAM Throughput                   %             0.93
    Duration                          s             6.07
    L1/TEX Cache Throughput           %            99.90
    L2 Cache Throughput               %             1.17
    SM Active Cycles              cycle 3,543,922,046.38
    Compute (SM) Throughput           %             6.05
    ----------------------- ----------- ----------------

    OPT   This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak
          performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak
          typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potential
          reasons.

    Section: Memory Workload Analysis
    ----------------- ----------- ------------
    Metric Name       Metric Unit Metric Value
    ----------------- ----------- ------------
    Memory Throughput     Gbyte/s         2.98
    Mem Busy                    %        49.95
    Max Bandwidth               %        13.85
    L1/TEX Hit Rate             %        97.32
    L2 Hit Rate                 %        92.61
    Mem Pipes Busy              %         6.05
    ----------------- ----------- ------------
```

The thing that stands out with the naive implementation is a poor utilization
of
the memory bandwidth (10.47%), which probably has to do with a poor access patterns. The coalescing kernel should be a remedy to that.

## Coalescing kernel

### Indswap

```sh
Running kernel 22 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000036) s, performance: (  115.1) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.000140) s, performance: (  239.4) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.000946) s, performance: (  283.9) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.005653) s, performance: (  379.9) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.037434) s, performance: (  458.9) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
Average elapsed time: (0.307420) s, performance: (  447.1) GFLOPS. size: (4096).

$ ncu --kernel-name sgemm_coalesce_indswap_dmalt \
    --launch-skip 256 --launch-count 1 \
    --section SpeedOfLight --section MemoryWorkloadAnalysis \
    ./sgemm/build/sgemm 22
==PROF== Connected to process 4812 (/content/sgemm/build/sgemm)
Running kernel 22 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000574) s, performance: (    7.3) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.000495) s, performance: (   67.8) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.000956) s, performance: (  280.9) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.005545) s, performance: (  387.3) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.037633) s, performance: (  456.5) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
==PROF== Profiling "sgemm_coalesce_indswap_dmalt": 0%....50%....100% - 8 passes
Average elapsed time: (0.432565) s, performance: (  317.7) GFLOPS. size: (4096).
==PROF== Disconnected from process 4812
[4812] sgemm@127.0.0.1
  sgemm_coalesce_indswap_dmalt(int, int, int, float, const float *, const float *, float, float *) (128, 128, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- --------------
    Metric Name             Metric Unit   Metric Value
    ----------------------- ----------- --------------
    DRAM Frequency                  Ghz           5.00
    SM Frequency                    Mhz         585.00
    Elapsed Cycles                cycle    276,609,087
    Memory Throughput                 %          77.68
    DRAM Throughput                   %           8.84
    Duration                         ms         472.84
    L1/TEX Cache Throughput           %          77.79
    L2 Cache Throughput               %           4.18
    SM Active Cycles              cycle 276,184,847.43
    Compute (SM) Throughput           %          77.68
    ----------------------- ----------- --------------

    INF   Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced.
          Check both the Compute Workload Analysis and Memory Workload Analysis sections.

    Section: Memory Workload Analysis
    ----------------- ----------- ------------
    Metric Name       Metric Unit Metric Value
    ----------------- ----------- ------------
    Memory Throughput     Gbyte/s        28.25
    Mem Busy                    %        38.84
    Max Bandwidth               %        77.68
    L1/TEX Hit Rate             %        94.98
    L2 Hit Rate                 %        49.56
    Mem Pipes Busy              %        77.68
    ----------------- ----------- ------------
```

### Flat block (Boehm's version)

```sh
``Running kernel 23 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000037) s, performance: (  113.6) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.000142) s, performance: (  236.7) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.000957) s, performance: (  280.6) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.005029) s, performance: (  427.0) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.034065) s, performance: (  504.3) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
Average elapsed time: (0.277539) s, performance: (  495.2) GFLOPS. size: (4096).`

$ ncu --kernel-name sgemm_coalesce_dmalt \
    --launch-skip 256 --launch-count 1 \
    --section SpeedOfLight --section MemoryWorkloadAnalysis \
    ./sgemm/build/sgemm 23
==PROF== Connected to process 6084 (/content/sgemm/build/sgemm)
Running kernel 23 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000374) s, performance: (   11.2) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.000410) s, performance: (   81.8) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.000965) s, performance: (  278.3) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.005730) s, performance: (  374.8) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.035210) s, performance: (  487.9) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
==PROF== Profiling "sgemm_coalesce_dmalt": 0%....50%....100% - 8 passes
Average elapsed time: (0.403874) s, performance: (  340.3) GFLOPS. size: (4096).
==PROF== Disconnected from process 6084
[6084] sgemm@127.0.0.1
  void sgemm_coalesce_dmalt<32>(int, int, int, float, const float *, const float *, float, float *) (128, 128, 1)x(1024, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- --------------
    Metric Name             Metric Unit   Metric Value
    ----------------------- ----------- --------------
    DRAM Frequency                  Ghz           5.00
    SM Frequency                    Mhz         585.00
    Elapsed Cycles                cycle    269,329,911
    Memory Throughput                 %          79.78
    DRAM Throughput                   %          12.05
    Duration                         ms         460.39
    L1/TEX Cache Throughput           %          79.87
    L2 Cache Throughput               %           4.30
    SM Active Cycles              cycle 268,970,023.02
    Compute (SM) Throughput           %          79.78
    ----------------------- ----------- --------------

    INF   Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced.
          Check both the Compute Workload Analysis and Memory Workload Analysis sections.

    Section: Memory Workload Analysis
    ----------------- ----------- ------------
    Metric Name       Metric Unit Metric Value
    ----------------- ----------- ------------
    Memory Throughput     Gbyte/s        38.51
    Mem Busy                    %        39.89
    Max Bandwidth               %        79.78
    L1/TEX Hit Rate             %        94.98
    L2 Hit Rate                 %        73.25
    Mem Pipes Busy              %        79.78
    ----------------- ----------- ------------
```

## Tiling

```sh
Running kernel 25 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000026) s, performance: (  162.9) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.000106) s, performance: (  315.5) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.000703) s, performance: (  381.6) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.003994) s, performance: (  537.7) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.019648) s, performance: (  874.4) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
Average elapsed time: (0.158643) s, performance: (  866.3) GFLOPS. size: (4096).
==PROF== Connected to process 7586 (/content/sgemm/build/sgemm)
Running kernel 25 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000470) s, performance: (    8.9) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.000405) s, performance: (   82.9) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.000712) s, performance: (  377.2) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.003883) s, performance: (  553.0) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.019745) s, performance: (  870.1) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
==PROF== Profiling "sgemm_shared_mem_block_dmalt": 0%....50%....100% - 8 passes
Average elapsed time: (0.257308) s, performance: (  534.1) GFLOPS. size: (4096).
==PROF== Disconnected from process 7586
[7586] sgemm@127.0.0.1
  void sgemm_shared_mem_block_dmalt<32>(int, int, int, float, const float *, const float *, float, float *) (128, 128, 1)x(1024, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- --------------
    Metric Name             Metric Unit   Metric Value
    ----------------------- ----------- --------------
    DRAM Frequency                  Ghz           5.00
    SM Frequency                    Mhz         585.00
    Elapsed Cycles                cycle    193,749,779
    Memory Throughput                 %          76.23
    DRAM Throughput                   %          12.90
    Duration                         ms         331.20
    L1/TEX Cache Throughput           %          90.41
    L2 Cache Throughput               %           5.97
    SM Active Cycles              cycle 193,488,364.88
    Compute (SM) Throughput           %          76.23
    ----------------------- ----------- --------------

    INF   Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced.
          Check both the Compute Workload Analysis and Memory Workload Analysis sections.

    Section: Memory Workload Analysis
    ----------------- ----------- ------------
    Metric Name       Metric Unit Metric Value
    ----------------- ----------- ------------
    Memory Throughput     Gbyte/s        41.24
    Mem Busy                    %        45.21
    Max Bandwidth               %        76.23
    L1/TEX Hit Rate             %         0.39
    L2 Hit Rate                 %        47.36
    Mem Pipes Busy              %        76.23
    ----------------- ----------- ------------

```

## 1D blocktiling

```sh
Running kernel 26 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000044) s, performance: (   95.3) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.000081) s, performance: (  413.7) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.000368) s, performance: (  728.6) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.002567) s, performance: (  836.6) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.009488) s, performance: ( 1810.6) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
Average elapsed time: (0.071322) s, performance: ( 1927.0) GFLOPS. size: (4096).

==PROF== Connected to process 1807 (/content/sgemm/build/sgemm)
Running kernel 26 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000486) s, performance: (    8.6) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.000512) s, performance: (   65.5) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.000531) s, performance: (  505.8) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.002575) s, performance: (  834.0) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.009757) s, performance: ( 1760.8) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
==PROF== Profiling "sgemm1DBlocktilingDmalt": 0%....50%....100% - 8 passes
Average elapsed time: (0.144498) s, performance: (  951.1) GFLOPS. size: (4096).
==PROF== Disconnected from process 1807
[1807] sgemm@127.0.0.1
  void sgemm1DBlocktilingDmalt<64, 64, 8, 8>(int, int, int, float, const float *, const float *, float, float *) (64, 64, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz          5.00
    SM Frequency                    Mhz        585.00
    Elapsed Cycles                cycle    86,897,823
    Memory Throughput                 %         54.12
    DRAM Throughput                   %         13.09
    Duration                         ms        148.54
    L1/TEX Cache Throughput           %         91.54
    L2 Cache Throughput               %          6.71
    SM Active Cycles              cycle 86,380,004.95
    Compute (SM) Throughput           %         54.12
    ----------------------- ----------- -------------

    OPT   This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak
          performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak
          typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potential
          reasons.

    Section: Memory Workload Analysis
    ----------------- ----------- ------------
    Metric Name       Metric Unit Metric Value
    ----------------- ----------- ------------
    Memory Throughput     Gbyte/s        41.85
    Mem Busy                    %        45.77
    Max Bandwidth               %        54.12
    L1/TEX Hit Rate             %         0.77
    L2 Hit Rate                 %        49.07
    Mem Pipes Busy              %        54.12
    ----------------- ----------- ------------
```

## 2D blocktiling

```sh
Running kernel 27 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000116) s, performance: (   36.3) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.000196) s, performance: (  170.9) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.000359) s, performance: (  748.5) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.001441) s, performance: ( 1490.0) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.005598) s, performance: ( 3069.1) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
Average elapsed time: (0.040126) s, performance: ( 3425.2) GFLOPS. size: (4096).

==PROF== Connected to process 3430 (/content/sgemm/build/sgemm)
Running kernel 27 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000439) s, performance: (    9.5) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.000377) s, performance: (   89.1) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.000378) s, performance: (  709.6) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.001598) s, performance: ( 1343.6) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.005671) s, performance: ( 3029.2) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
==PROF== Profiling "sgemm2DBlocktilingDmalt": 0%....50%....100% - 8 passes
Average elapsed time: (0.099048) s, performance: ( 1387.6) GFLOPS. size: (4096).
==PROF== Disconnected from process 3430
[3430] sgemm@127.0.0.1
  void sgemm2DBlocktilingDmalt<128, 128, 8, 8, 8>(int, int, int, float, const float *, const float *, float, float *) (32, 32, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz          4.99
    SM Frequency                    Mhz        585.00
    Elapsed Cycles                cycle    44,701,799
    Memory Throughput                 %         48.53
    DRAM Throughput                   %         12.76
    Duration                         ms         76.41
    L1/TEX Cache Throughput           %         97.07
    L2 Cache Throughput               %          7.22
    SM Active Cycles              cycle 43,939,572.20
    Compute (SM) Throughput           %         61.37
    ----------------------- ----------- -------------

    OPT   Compute is more heavily utilized than Memory: Look at the Compute Workload Analysis section to see what the
          compute pipelines are spending their time doing. Also, consider whether any computation is redundant and
          could be reduced or moved to look-up tables.

    Section: Memory Workload Analysis
    ----------------- ----------- ------------
    Metric Name       Metric Unit Metric Value
    ----------------- ----------- ------------
    Memory Throughput     Gbyte/s        40.76
    Mem Busy                    %        48.53
    Max Bandwidth               %        26.39
    L1/TEX Hit Rate             %        18.80
    L2 Hit Rate                 %        57.42
    Mem Pipes Busy              %        26.39
    ----------------- ----------- ------------
```

## Vectorize

```sh
Running kernel 28 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000094) s, performance: (   44.8) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.000172) s, performance: (  195.4) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.000328) s, performance: (  819.0) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.001346) s, performance: ( 1595.3) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.005472) s, performance: ( 3139.7) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
Average elapsed time: (0.036921) s, performance: ( 3722.5) GFLOPS. size: (4096).
==PROF== Connected to process 4925 (/content/sgemm/build/sgemm)
Running kernel 28 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000431) s, performance: (    9.7) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.000397) s, performance: (   84.5) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.000576) s, performance: (  465.7) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.001431) s, performance: ( 1500.2) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.005443) s, performance: ( 3156.3) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
==PROF== Profiling "sgemmVectorizeDmalt": 0%....50%....100% - 8 passes
Average elapsed time: (0.095558) s, performance: ( 1438.3) GFLOPS. size: (4096).
==PROF== Disconnected from process 4925
[4925] sgemm@127.0.0.1
  void sgemmVectorizeDmalt<128, 128, 8, 8, 8>(int, int, int, float, float *, float *, float, float *) (32, 32, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz          5.00
    SM Frequency                    Mhz        585.00
    Elapsed Cycles                cycle    41,577,791
    Memory Throughput                 %         48.76
    DRAM Throughput                   %         10.27
    Duration                         ms         71.07
    L1/TEX Cache Throughput           %         97.52
    L2 Cache Throughput               %          7.22
    SM Active Cycles              cycle 40,886,968.58
    Compute (SM) Throughput           %         65.22
    ----------------------- ----------- -------------

    OPT   Compute is more heavily utilized than Memory: Look at the Compute Workload Analysis section to see what the
          compute pipelines are spending their time doing. Also, consider whether any computation is redundant and
          could be reduced or moved to look-up tables.

    Section: Memory Workload Analysis
    ----------------- ----------- ------------
    Metric Name       Metric Unit Metric Value
    ----------------- ----------- ------------
    Memory Throughput     Gbyte/s        32.83
    Mem Busy                    %        48.76
    Max Bandwidth               %        26.32
    L1/TEX Hit Rate             %         4.42
    L2 Hit Rate                 %        60.10
    Mem Pipes Busy              %        19.70
    ----------------- ----------- ------------
```

## Comparison

| Kernel name                  | 4096 GFLOPS | Mem TP 1 | DRAM TP | L1 TP | L2 TP | Compute TP | Mem TP 2 | Mem Busy | Max BW | L1 HR | L2 HR | Mem Pipes Busy |
| ---------------------------- | ----------- | -------- | ------- | ----- | ----- | ---------- | -------- | -------- | ------ | ----- | ----- | -------------- |
| sgemm_naive_dmalt            | 61.4        | 49.95    | 0.93    | 99.9  | 1.17  | 6.05       | 2.98     | 49.95    | 13.85  | 97.32 | 92.61 | 6.05           |
| sgemm_coalesce_indswap_dmalt | 447.1       | 77.68    | 8.84    | 77.79 | 4.18  | 77.68      | 28.25    | 38.84    | 77.68  | 94.98 | 49.56 | 77.68          |
| sgemm_coalesce_dmalt         | 495.2       | 79.78    | 12.05   | 79.87 | 4.30  | 79.78      | 38.51    | 39.89    | 79.78  | 94.98 | 73.25 | 79.78          |
| sgemm_shared_mem_block_dmalt | 866.3       | 76.23    | 12.90   | 90.41 | 5.97  | 76.23      | 41.24    | 45.21    | 76.23  | 0.39  | 47.36 | 76.23          |
| gemm_1d_blocktiling          | 1927.0      | 54.1     | 13.09   | 91.54 | 6.71  | 54.12      | 41.85    | 45.77    | 54.12  | 0.77  | 49.07 | 54.12          |
| gemm_2d_blocktiling          | 3425.2      | 48.53    | 12.76   | 97.07 | 7.22  | 61.37      | 40.76    | 48.53    | 26.39  | 18.80 | 57.42 | 26.39          |
| gemm_vectorize               | 3722.5      | 48.76    | 10.27   | 97.52 | 7.22  | 65.22      | 32.83    | 48.76    | 26.32  | 4.42  | 60.10 | 19.70          |

## Notes

`sgemm_coalesce_indswap_dmalt` vs `sgemm_coalesce_dmalt` shenanigan:

- Both kernels are byte-for-byte identical at L1 and L2 request level
- They differ only in DRAM bytes per L2 miss (48.8 vs 122.5)
- Neither difference reaches performance: both are load-issue bound at ~79%, and across
  six paired runs the throughput gap is 2.5%, inside run-to-run variance

`sgemm_shared_mem_block_dmalt`

Tiling was supposed to reduce the number of the DRAM accesses from 4096**3 *2
to 32* 4096 *2* (4096 / 32)** 2 = 4096 \**3 *2 / 32, so x32. However, we
don't see it from the metrics. Mem TP 2 (Memory Throughput from the Memory
Workload Analysis table) shows 41.24 GB/s* 331 ms = 13.65 GB of DRAM vs 38.51
GB/s* 460 ms = 17.71 GB for the coalescing kernel, so the DRAM traffic is
indeed reduced, but not 32-fold -- only by 23%. The suspected reason that L1
cache absorbs the hit in the coalescing kernel.
