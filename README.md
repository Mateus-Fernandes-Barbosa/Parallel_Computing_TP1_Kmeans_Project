# Execução e requisitos (OpenMP + MPI)

Projeto de programação paralela do algoritmo Kmeans. Implementações:
- Kmeans_sequencial. [Link para o código original](https://github.com/marcoscastro/kmeans/blob/master/kmeans.cpp)
- kmeans_openmp. Implementado somente com as diretivas do pragma openmp.
- kmeans_mpi_openmp. Estratégia híbrida com MPI e OpenMp.

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
