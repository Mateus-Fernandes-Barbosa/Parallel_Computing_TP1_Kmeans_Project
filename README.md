# Execução e requisitos (OpenMP + MPI + GPU)

Projeto de programação paralela do algoritmo Kmeans. Implementações:
- Kmeans_sequencial. [Link para o código original](https://github.com/marcoscastro/kmeans/blob/master/kmeans.cpp)
- kmeans_openmp. Implementado somente com as diretivas do pragma openmp.
- kmeans_mpi_openmp. Estratégia híbrida com MPI e OpenMp.
- kmeans_openmp_GPU. Implementação com OpenMP Offloading para GPU.
- kmeans_cuda. Implementação com CUDA para GPU (variação de threads por bloco: 128, 256, 512).

## Requisitos e instruções por sistema

<details>
<summary>Instruções para Windows</summary>

#### Requisitos:
- Compilador: GCC (MinGW/WSL) com suporte a -fopenmp.
- MPI: MS-MPI (instalar SDK).

#### Instalação (exemplo):
1. Instalar MinGW ou usar WSL com GCC.
2. Baixar e instalar [MS-MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467).
3. Configurar variáveis de ambiente (adapte conforme sua instalação):
    - setx MSMPI_INC "C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
    - setx MSMPI_LIB64 "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"

#### Utilização dos scripts de teste:
Executar o `benchmark.bat` para compilação e testes automáticos com controle de tempo de execução.

#### Alternativa para execução manual:
##### Compilação:
- Sequencial:
  ```cmd
  g++ -std=c++17 -o kmeans_sequencial kmeans_sequencial.cpp
  ```
- OpenMP:
  ```cmd
  g++ -std=c++17 -fopenmp -o kmeans_openmp kmeans_openmp.cpp
  ```
- MPI + OpenMP (MS-MPI):
  ```cmd
  g++ -std=c++17 -fopenmp -I"%MSMPI_INC%" -L"%MSMPI_LIB64%" -lmsmpi -o kmeans_mpi_openmp kmeans_mpi_openmp.cpp
  ```

##### Execução:
- Sequencial:
  ```cmd
  .\kmeans_sequencial < UCI_Credit_Card.txt
  ```
- OpenMP (definir threads):
  ```cmd
  set OMP_NUM_THREADS=4
  .\kmeans_openmp < UCI_Credit_Card.txt
  ```
- MPI + OpenMP (definir OMP_NUM_THREADS e processos):
  ```cmd
  set OMP_NUM_THREADS=2
  mpiexec -n 4 .\kmeans_mpi_openmp < UCI_Credit_Card.txt
  ```
</details>

<details>
<summary>Instruções para Linux (Debian/Ubuntu)</summary>

#### Requisitos:
- Compilador: GCC com suporte a -fopenmp.
- MPI: OpenMPI ou MPICH.

#### Instalação (exemplo):
```bash
sudo apt update
sudo apt install build-essential g++ libopenmpi-dev openmpi-bin
```

#### Utilização dos scripts de teste:
Executar o `benchmark.sh` para compilação e testes automáticos com controle de tempo de execução.

#### Alternativa para execução manual:
##### Compilação:
- Sequencial:
  ```bash
  g++ -std=c++17 -o kmeans_sequencial kmeans_sequencial.cpp
  ```
- OpenMP:
  ```bash
  g++ -std=c++17 -fopenmp -o kmeans_openmp kmeans_openmp.cpp
  ```
- MPI + OpenMP:
  ```bash
  mpicxx -std=c++17 -fopenmp -o kmeans_mpi_openmp kmeans_mpi_openmp.cpp
  ```

##### Execução:
- Sequencial:
  ```bash
  ./kmeans_sequencial < UCI_Credit_Card.txt
  ```
- OpenMP (definir threads):
  ```bash
  export OMP_NUM_THREADS=4
  ./kmeans_openmp < UCI_Credit_Card.txt
  ```
- MPI + OpenMP (definir OMP_NUM_THREADS e processos):
  ```bash
  export OMP_NUM_THREADS=2
  mpiexec -n 4 ./kmeans_mpi_openmp < UCI_Credit_Card.txt
  ```
</details>

<details>
<summary>Instruções para GPU (Windows)</summary>

#### Requisitos:
- Compilador: GCC (MinGW/WSL) com suporte a -fopenmp.
- CUDA: Toolkit NVIDIA (para kmeans_cuda).

#### Instalação (exemplo):
1. Instalar [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (requer GPU NVIDIA).
2. Adicionar `nvcc` ao PATH (geralmente: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`).
3. Verificar instalação: `nvcc --version`

#### Compilação:
- OpenMP + GPU (offload):
  ```cmd
  g++ -std=c++17 -fopenmp -foffload=nvptx-none -o kmeans_openmp_GPU kmeans_openmp_GPU.cpp
  ```
- CUDA (blocos automáticos; pode variar threads via parâmetro):
  ```cmd
  nvcc -O0 -o kmeans_cuda kmeans_cuda.cu
  ```

#### Execução:
- OpenMP + GPU (definir threads):
  ```cmd
  set OMP_NUM_THREADS=4
  .\kmeans_openmp_GPU < UCI_Credit_Card.txt > output.txt
  ```
- CUDA (com variação de threads; 256 threads/bloco por padrão):
  ```cmd
  .\kmeans_cuda output.txt 256 < UCI_Credit_Card.txt
  ```
  Sintaxe: `kmeans_cuda <output> [gpu_threads] < input.txt` (blocos são calculados automaticamente)
</details>

<details>
<summary>Instruções para GPU (Linux)</summary>

#### Requisitos:
- Compilador: GCC com suporte a -fopenmp.
- CUDA: Toolkit NVIDIA (para kmeans_cuda).

#### Instalação (exemplo):
```bash
sudo apt update
# Instalar CUDA Toolkit (verificar versão específica para sua distribuição)
sudo apt install nvidia-cuda-toolkit
# Verificar instalação
nvcc --version
```

#### Compilação:
- OpenMP + GPU (offload):
  ```bash
  g++ -std=c++17 -fopenmp -foffload=nvptx-none -o kmeans_openmp_GPU kmeans_openmp_GPU.cpp -lm
  ```
- CUDA (blocos automáticos; pode variar threads via parâmetro):
  ```bash
  nvcc -O0 -o kmeans_cuda kmeans_cuda.cu
  ```

#### Execução:
- OpenMP + GPU (definir threads):
  ```bash
  export OMP_NUM_THREADS=4
  ./kmeans_openmp_GPU < UCI_Credit_Card.txt > output.txt
  ```
- CUDA (com variação de threads; 256 threads/bloco por padrão):
  ```bash
  ./kmeans_cuda output.txt 256 < UCI_Credit_Card.txt
  ```
  Sintaxe: `kmeans_cuda <output> [gpu_threads] < input.txt` (blocos são calculados automaticamente)
</details>

## Benchmark automático

### Linux: benchmark.sh
Executa todas as variações (Sequencial, OpenMP, MPI+OpenMP, OpenMP+GPU, CUDA com diferentes threads):
```bash
./benchmark.sh UCI_Credit_Card.txt
```
As thread counts CUDA testadas são: **128, 256, 512**.

### Windows: benchmark.bat
Executa variações de Sequencial, OpenMP e MPI+OpenMP:
```cmd
.\benchmark.bat
```
**Nota:** GPU não está integrada ao benchmark.bat no momento. Use comandos manuais para testar CUDA e OpenMP+GPU.
