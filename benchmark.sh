#!/bin/bash
# benchmark2.sh - Benchmarks: Sequencial, OpenMP, MPI+OpenMP (works1) e teste2

# ==============================
# Estilo/cores (igual ao benchmark.sh)
# ==============================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ==============================
# Entrada e diretórios
# ==============================
RAW_INPUT=${1:-UCI_Credit_Card.txt}
# Tenta ajustar caminho automaticamente se o arquivo não estiver na raiz
if [ ! -f "$RAW_INPUT" ] && [ -f "databases/$RAW_INPUT" ]; then
    INPUT_FILE="databases/$RAW_INPUT"
else
    INPUT_FILE="$RAW_INPUT"
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}❌ Erro: Arquivo de entrada '$INPUT_FILE' não encontrado.${NC}"
    echo -e "${YELLOW}Dica:${NC} passe um caminho válido ou coloque o arquivo em ./databases/"
    exit 1
fi

OUTPUT_DIR="./outputs"
mkdir -p "$OUTPUT_DIR"

# ==============================
# Configurações de execução (EDITÁVEIS)
# ==============================
# OpenMP threads (apenas threads, sem MPI)
OPENMP_THREADS=(1 2 4 8)
# CUDA launch params to test: threads per block and explicit block counts (0 = auto compute)
CUDA_THREADS=(2 4 8 16 32 64 128 256 512 1024)
# explicit block counts removed; blocks are computed automatically inside the binary
# kmeans_mpi_openmp: arrays de processos e threads para testar diferentes combinações
MPI_OPENMP_PROCESSES=(1 1 1 2 4)
MPI_OPENMP_THREADS=(1 2 4 2 1)
# Exemplo: executa (1p,1t), (1p,2t), (1p,4t), (2p,2t), (4p,1t)

# ==============================
# Banners
# ==============================
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      BENCHMARK K-MEANS - SEQ | OMP | MPI+OMP           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo -e "${YELLOW}📁 Arquivo de entrada:${NC} $INPUT_FILE"

# ==============================
# Compilação (na ordem solicitada)
# ==============================
echo -e "\n${YELLOW}🔨 Compilando binários...${NC}"

echo -e "  • ${BLUE}1) kmeans_sequencial${NC}"
g++ -std=c++17 -O0 -o kmeans_sequencial kmeans_sequencial.cpp -lm || { echo -e "${RED}❌ Falha ao compilar kmeans_sequencial.cpp${NC}"; exit 1; }

echo -e "  • ${BLUE}2) kmeans_openmp${NC}"
g++ -std=c++17 -O0 -fopenmp -o kmeans_openmp kmeans_openmp.cpp -lm || { echo -e "${RED}❌ Falha ao compilar kmeans_openmp.cpp${NC}"; exit 1; }

echo -e "  • ${BLUE}3) kmeans_mpi_openmp (MPI+OpenMP)${NC}"
mpicxx -std=c++17 -O0 -fopenmp -o kmeans_mpi_openmp kmeans_mpi_openmp.cpp -lm || { echo -e "${RED}❌ Falha ao compilar kmeans_mpi_openmp.cpp${NC}"; exit 1; }

echo -e "  • ${BLUE}3) kmeans_openmp_GPU (OpenMp com gpu)${NC}"
g++ -std=c++17 -O0 -fopenmp -o kmeans_openmp_GPU kmeans_openmp_GPU.cpp -lm || { echo -e "${RED}❌ Falha ao compilar kmeans_openmp_GPU.cpp${NC}"; exit 1; }

echo -e "  • ${BLUE}4) kmeans_cuda (CUDA)${NC}"
nvcc -O0 -o kmeans_cuda kmeans_cuda.cu -arch=sm_75 || { echo -e "${RED}❌ Falha ao compilar kmeans_cuda.cu${NC}"; exit 1; }

echo -e "${GREEN}   ✅ Compilação OK${NC}"

# ==============================
# Funções auxiliares
# ==============================
measure_time() {
    local CMD=$1
    local START END
    START=$(date +%s.%N)
    eval "$CMD"
    local STATUS=$?
    END=$(date +%s.%N)
    local DIFF=$(echo "$END - $START" | bc)
    echo "$DIFF $STATUS"
}

# Compara centróides (linhas "Cluster values:") entre 2 arquivos
compare_centroids() {
    local A=$1
    local B=$2
    local TMPA="$OUTPUT_DIR/.cmp_$(basename "$A").txt"
    local TMPB="$OUTPUT_DIR/.cmp_$(basename "$B").txt"
    grep "Cluster values:" "$A" | sort > "$TMPA" 2>/dev/null
    grep "Cluster values:" "$B" | sort > "$TMPB" 2>/dev/null
    if diff -q "$TMPA" "$TMPB" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

highlight_check() {
    local OK=$1
    local LABEL=$2
    if [ "$OK" -eq 0 ]; then
        echo -e "    ${GREEN}✅ $LABEL${NC}"
    else
        echo -e "    ${RED}❌ $LABEL${NC}"
    fi
}

# Compara um arquivo (LABEL) com as baselines (sequencial e OpenMP 1 thread)
check_against_baselines() {
    local FILE=$1
    local LABEL=$2
    echo -e "${BLUE}• $LABEL${NC}"
    compare_centroids "$FILE" "$SEQ_OUT"; local A=$?
    compare_centroids "$FILE" "$OMP1_OUT"; local B=$?
    highlight_check $A "Comparação com Sequencial"
    highlight_check $B "Comparação com OpenMP (1 thread)"
}

# ==============================
# Execuções
# ==============================
declare -A TIMES

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}⏱️  Executando SEQUENCIAL...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
SEQ_OUT="$OUTPUT_DIR/out_seq.txt"
read TIME_SEQ STATUS < <(measure_time "./kmeans_sequencial \"$SEQ_OUT\" < \"$INPUT_FILE\"")
TIMES[seq]="$TIME_SEQ"
if [ "$STATUS" -ne 0 ]; then echo -e "${RED}❌ Falha na execução sequencial${NC}"; exit 1; fi
echo -e "${GREEN}✅ Concluído em: ${TIME_SEQ}s${NC}"

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}⏱️  Executando OpenMP (${OPENMP_THREADS[@]} threads)...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
for T in "${OPENMP_THREADS[@]}"; do
    OMP_OUT="$OUTPUT_DIR/out_openmp_${T}t.txt"
    echo -e "${YELLOW}   🔧 ${T} thread(s)${NC}"
    read TTIME STATUS < <(measure_time "OMP_NUM_THREADS=${T} ./kmeans_openmp \"$OMP_OUT\" < \"$INPUT_FILE\"")
    TIMES[omp_${T}]="$TTIME"
    if [ "$STATUS" -ne 0 ]; then echo -e "${RED}   ❌ Falha no OpenMP (${T}t)${NC}"; exit 1; fi
    echo -e "${GREEN}   ✅ Tempo: ${TTIME}s${NC}"
done

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}⏱️  Executando kmeans_mpi_openmp (MPI+OpenMP)...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
for i in "${!MPI_OPENMP_PROCESSES[@]}"; do
    P=${MPI_OPENMP_PROCESSES[$i]}
    T=${MPI_OPENMP_THREADS[$i]}
    MPI_OPENMP_OUT="$OUTPUT_DIR/out_mpi_openmp_${P}p_${T}t.txt"
    echo -e "${YELLOW}   🔧 ${P} processo(s), ${T} thread(s)${NC}"
    read TTIME STATUS < <(measure_time "OMP_NUM_THREADS=${T} mpirun -np ${P} ./kmeans_mpi_openmp \"$MPI_OPENMP_OUT\" < \"$INPUT_FILE\"")
    TIMES[mpi_openmp_${P}p_${T}t]="$TTIME"
    if [ "$STATUS" -ne 0 ]; then echo -e "${RED}   ❌ Falha no kmeans_mpi_openmp (${P}p, ${T}t)${NC}"; exit 1; fi
    echo -e "${GREEN}   ✅ Tempo: ${TTIME}s${NC}"
done

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}⏱️  Executando kmeans_openmp_GPU (OpenMP com GPU)...${NC}"
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
for T in "${OPENMP_THREADS[@]}"; do
    OMP_GPU_OUT="$OUTPUT_DIR/out_openmp_GPU_${T}t.txt"
    echo -e "${YELLOW}   🔧 ${T} thread(s)${NC}"
    read TTIME STATUS < <(measure_time "OMP_NUM_THREADS=${T} ./kmeans_openmp_GPU \"$OMP_GPU_OUT\" < \"$INPUT_FILE\"")
    TIMES[omp_gpu_${T}]="$TTIME"
    if [ "$STATUS" -ne 0 ]; then echo -e "${RED}   ❌ Falha no kmeans_openmp_GPU (${T}t)${NC}"; exit 1; fi
    echo -e "${GREEN}   ✅ Tempo: ${TTIME}s${NC}"
done

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}⏱️  Executando kmeans_cuda (GPU) com variações de threads/blocos...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
for T in "${CUDA_THREADS[@]}"; do
    CUDA_OUT="$OUTPUT_DIR/out_cuda_t${T}.txt"
    echo -e "${YELLOW}   🔧 threads=${T}${NC}"
    read TTIME STATUS < <(measure_time "./kmeans_cuda \"$CUDA_OUT\" ${T} < \"$INPUT_FILE\"")
    TIMES[cuda_t${T}]="$TTIME"
    if [ "$STATUS" -ne 0 ]; then echo -e "${RED}   ❌ Falha na execução kmeans_cuda (t=${T})${NC}"; exit 1; fi
    echo -e "${GREEN}   ✅ Tempo: ${TTIME}s${NC}"
done

# ==============================
# Verificação de corretude
# ==============================
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}🔍 Verificando corretude (centróides)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Baselines: sequencial e OpenMP com 1 thread
SEQ_OUT="$OUTPUT_DIR/out_seq.txt"
OMP1_OUT="$OUTPUT_DIR/out_openmp_1t.txt"

# Comparar omp 1t com seq
echo -e "${BLUE}• OpenMP (1 thread) vs Sequencial${NC}"
compare_centroids "$OMP1_OUT" "$SEQ_OUT"; highlight_check $? "Centroides idênticos (omp 1t ↔ seq)"

# Checar OpenMP: comparar todos exceto o primeiro com Sequencial e OpenMP (1 thread)
for T in "${OPENMP_THREADS[@]:1}"; do  # pula o primeiro elemento
    check_against_baselines "$OUTPUT_DIR/out_openmp_${T}t.txt" "OpenMP (${T} threads)"
done

# Checar kmeans_openmp_GPU: comparar todas as configurações (inclui 1 thread)
for T in "${OPENMP_THREADS[@]}"; do
    check_against_baselines "$OUTPUT_DIR/out_openmp_GPU_${T}t.txt" "kmeans_openmp_GPU (${T} threads)"
done

# Checar kmeans_mpi_openmp: comparar todas as configurações exceto a primeira
MPI_OPENMP_BASELINE_P=${MPI_OPENMP_PROCESSES[0]}
MPI_OPENMP_BASELINE_T=${MPI_OPENMP_THREADS[0]}
MPI_OPENMP_BASELINE_FILE="$OUTPUT_DIR/out_mpi_openmp_${MPI_OPENMP_BASELINE_P}p_${MPI_OPENMP_BASELINE_T}t.txt"

for i in "${!MPI_OPENMP_PROCESSES[@]}"; do
    if [ $i -eq 0 ]; then continue; fi  # pula baseline (primeira configuração)
    P=${MPI_OPENMP_PROCESSES[$i]}
    T=${MPI_OPENMP_THREADS[$i]}
    FILE="$OUTPUT_DIR/out_mpi_openmp_${P}p_${T}t.txt"
    echo -e "${BLUE}• kmeans_mpi_openmp (${P}p, ${T}t)${NC}"
    compare_centroids "$FILE" "$SEQ_OUT"; highlight_check $? "Comparação com Sequencial"
    compare_centroids "$FILE" "$MPI_OPENMP_BASELINE_FILE"; highlight_check $? "Comparação com mpi_openmp baseline (${MPI_OPENMP_BASELINE_P}p, ${MPI_OPENMP_BASELINE_T}t)"
done

# Checar kmeans_cuda (GPU) — todas as variações geradas
for T in "${CUDA_THREADS[@]}"; do
    FILE="$OUTPUT_DIR/out_cuda_t${T}.txt"
    check_against_baselines "$FILE" "kmeans_cuda (t=${T})"
done

# ==============================
# Resumo simples de tempos
# ==============================
echo -e "\n${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  RESUMO DE TEMPOS (s)                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
printf "%-36s %s\n" "Sequencial" "${TIMES[seq]}"
for T in "${OPENMP_THREADS[@]}"; do 
    printf "%-36s %s\n" "OpenMP (${T}t)" "${TIMES[omp_${T}]}"
done

# Mostrar tempos do OpenMP com GPU
for T in "${OPENMP_THREADS[@]}"; do
    printf "%-36s %s\n" "OpenMP+GPU (${T}t)" "${TIMES[omp_gpu_${T}]}"
done

for i in "${!MPI_OPENMP_PROCESSES[@]}"; do
    P=${MPI_OPENMP_PROCESSES[$i]}
    T=${MPI_OPENMP_THREADS[$i]}
    printf "%-36s %s\n" "kmeans_mpi_openmp (${P}p, ${T}t)" "${TIMES[mpi_openmp_${P}p_${T}t]}"
done

# Mostrar tempos do kmeans_cuda (t,b)
for T in "${CUDA_THREADS[@]}"; do
    key="cuda_t${T}"
    printf "%-36s %s\n" "kmeans_cuda (t=${T})" "${TIMES[$key]}"
done

# ==============================
# Speedup vs Sequencial
# ==============================
echo -e "\n${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                     RESUMO DE SPEEDUP                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"

SEQ_T="${TIMES[seq]}"
calc_speedup() {
    local base=$1; local t=$2
    if [ -z "$base" ] || [ -z "$t" ]; then echo "-"; return; fi
    echo "scale=4; $base / $t" | bc -l
}

best_label="Sequencial"
best_speed=1.0

printf "%-36s %s\n" "Sequencial" "1.00x"
for T in "${OPENMP_THREADS[@]}"; do
    sp=$(calc_speedup "$SEQ_T" "${TIMES[omp_${T}]}")
    printf "%-36s %s\n" "OpenMP (${T} threads)" "${sp}x"
    if [ "$sp" != "-" ] && echo "$sp > $best_speed" | bc -l >/dev/null 2>&1; then best_speed=$sp; best_label="OpenMP (${T} threads)"; fi
done

# Speedup para OpenMP com GPU
for T in "${OPENMP_THREADS[@]}"; do
    sp=$(calc_speedup "$SEQ_T" "${TIMES[omp_gpu_${T}]}")
    printf "%-36s %s\n" "OpenMP+GPU (${T} threads)" "${sp}x"
    if [ "$sp" != "-" ] && echo "$sp > $best_speed" | bc -l >/dev/null 2>&1; then best_speed=$sp; best_label="OpenMP+GPU (${T} threads)"; fi
done

for i in "${!MPI_OPENMP_PROCESSES[@]}"; do
    P=${MPI_OPENMP_PROCESSES[$i]}
    T=${MPI_OPENMP_THREADS[$i]}
    sp=$(calc_speedup "$SEQ_T" "${TIMES[mpi_openmp_${P}p_${T}t]}")
    printf "%-36s %s\n" "kmeans_mpi_openmp (${P}p, ${T}t)" "${sp}x"
    if [ "$sp" != "-" ] && echo "$sp > $best_speed" | bc -l >/dev/null 2>&1; then best_speed=$sp; best_label="kmeans_mpi_openmp (${P}p, ${T}t)"; fi
done

for T in "${CUDA_THREADS[@]}"; do
    key="cuda_t${T}"
    sp=$(calc_speedup "$SEQ_T" "${TIMES[$key]}")
    printf "%-36s %s\n" "kmeans_cuda (t=${T})" "${sp}x"
    if [ "$sp" != "-" ] && echo "$sp > $best_speed" | bc -l >/dev/null 2>&1; then best_speed=$sp; best_label="kmeans_cuda (t=${T})"; fi
done

echo -e "\n${BLUE}🏆 Melhor resultado:${NC} ${GREEN}${best_speed}x${NC} com ${YELLOW}${best_label}${NC}"

echo -e "\n${GREEN}Saídas salvas em:${NC} $OUTPUT_DIR"