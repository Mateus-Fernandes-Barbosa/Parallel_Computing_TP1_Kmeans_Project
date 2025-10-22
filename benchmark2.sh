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

# works1: arrays de processos e threads para testar diferentes combinações
WORKS1_PROCESSES=(1 1 1 2 4)
WORKS1_THREADS=(1 2 4 2 1)
# Exemplo: executa (1p,1t), (1p,2t), (1p,4t), (2p,2t), (4p,1t)

# teste2: arrays de processos e threads para testar diferentes combinações
TESTE2_PROCESSES=(1 1 1 2 4)
TESTE2_THREADS=(1 2 4 2 1)
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
g++ -std=c++17 -O2 -o kmeans_sequencial kmeans_sequencial.cpp -lm || { echo -e "${RED}❌ Falha ao compilar kmeans_sequencial.cpp${NC}"; exit 1; }

echo -e "  • ${BLUE}2) kmeans_openmp${NC}"
g++ -std=c++17 -fopenmp -O2 -o kmeans_openmp kmeans_openmp.cpp -lm || { echo -e "${RED}❌ Falha ao compilar kmeans_openmp.cpp${NC}"; exit 1; }

echo -e "  • ${BLUE}3) kmeans_openmp_mpi_works1${NC}"
mpicxx -std=c++17 -fopenmp -O2 -o kmeans_openmp_mpi_works1 kmeans_openmp_mpi_works1.cpp -lm || { echo -e "${RED}❌ Falha ao compilar kmeans_openmp_mpi_works1.cpp${NC}"; exit 1; }

echo -e "  • ${BLUE}4) teste2 (MPI+OpenMP)${NC}"
mpicxx -std=c++17 -fopenmp -O2 -o teste2 teste2.cpp -lm || { echo -e "${RED}❌ Falha ao compilar teste2.cpp${NC}"; exit 1; }

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
echo -e "${YELLOW}⏱️  Executando MPI+OpenMP (kmeans_openmp_mpi_works1)...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
for i in "${!WORKS1_PROCESSES[@]}"; do
    P=${WORKS1_PROCESSES[$i]}
    T=${WORKS1_THREADS[$i]}
    MPIW1_OUT="$OUTPUT_DIR/out_mpi_works1_${P}p_${T}t.txt"
    echo -e "${YELLOW}   🔧 ${P} processo(s), ${T} thread(s)${NC}"
    read TTIME STATUS < <(measure_time "OMP_NUM_THREADS=${T} mpirun -np ${P} ./kmeans_openmp_mpi_works1 \"$MPIW1_OUT\" < \"$INPUT_FILE\"")
    TIMES[mpiworks1_${P}p_${T}t]="$TTIME"
    if [ "$STATUS" -ne 0 ]; then echo -e "${RED}   ❌ Falha no MPI+OpenMP works1 (${P}p, ${T}t)${NC}"; exit 1; fi
    echo -e "${GREEN}   ✅ Tempo: ${TTIME}s${NC}"
done

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}⏱️  Executando teste2 (MPI+OpenMP)...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
for i in "${!TESTE2_PROCESSES[@]}"; do
    P=${TESTE2_PROCESSES[$i]}
    T=${TESTE2_THREADS[$i]}
    T2_OUT="$OUTPUT_DIR/out_teste2_${P}p_${T}t.txt"
    echo -e "${YELLOW}   🔧 ${P} processo(s), ${T} thread(s)${NC}"
    read TTIME STATUS < <(measure_time "OMP_NUM_THREADS=${T} mpirun -np ${P} ./teste2 \"$T2_OUT\" < \"$INPUT_FILE\"")
    TIMES[teste2_${P}p_${T}t]="$TTIME"
    if [ "$STATUS" -ne 0 ]; then echo -e "${RED}   ❌ Falha no teste2 (${P}p, ${T}t)${NC}"; exit 1; fi
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

# Função para testar um arquivo contra ambos baselines
check_against_baselines() {
    local FILE=$1
    local LABEL=$2
    echo -e "${BLUE}• $LABEL${NC}"
    compare_centroids "$FILE" "$SEQ_OUT"; local A=$?
    compare_centroids "$FILE" "$OMP1_OUT"; local B=$?
    highlight_check $A "Comparação com Sequencial"
    highlight_check $B "Comparação com OpenMP (1 thread)"
}

# Checar OpenMP: comparar todos exceto o primeiro com Sequencial e OpenMP (1 thread)
for T in "${OPENMP_THREADS[@]:1}"; do  # pula o primeiro elemento
    check_against_baselines "$OUTPUT_DIR/out_openmp_${T}t.txt" "OpenMP (${T} threads)"
done

# Checar works1: comparar todas as configurações exceto a primeira
WORKS1_BASELINE_P=${WORKS1_PROCESSES[0]}
WORKS1_BASELINE_T=${WORKS1_THREADS[0]}
WORKS1_BASELINE_FILE="$OUTPUT_DIR/out_mpi_works1_${WORKS1_BASELINE_P}p_${WORKS1_BASELINE_T}t.txt"

for i in "${!WORKS1_PROCESSES[@]}"; do
    if [ $i -eq 0 ]; then continue; fi  # pula baseline (primeira configuração)
    P=${WORKS1_PROCESSES[$i]}
    T=${WORKS1_THREADS[$i]}
    FILE="$OUTPUT_DIR/out_mpi_works1_${P}p_${T}t.txt"
    echo -e "${BLUE}• MPI+OpenMP works1 (${P}p, ${T}t)${NC}"
    compare_centroids "$FILE" "$SEQ_OUT"; highlight_check $? "Comparação com Sequencial"
    compare_centroids "$FILE" "$WORKS1_BASELINE_FILE"; highlight_check $? "Comparação com works1 baseline (${WORKS1_BASELINE_P}p, ${WORKS1_BASELINE_T}t)"
done

# Checar teste2: comparar todas as configurações exceto a primeira  
TESTE2_BASELINE_P=${TESTE2_PROCESSES[0]}
TESTE2_BASELINE_T=${TESTE2_THREADS[0]}
TESTE2_BASELINE_FILE="$OUTPUT_DIR/out_teste2_${TESTE2_BASELINE_P}p_${TESTE2_BASELINE_T}t.txt"

for i in "${!TESTE2_PROCESSES[@]}"; do
    if [ $i -eq 0 ]; then continue; fi  # pula baseline (primeira configuração)
    P=${TESTE2_PROCESSES[$i]}
    T=${TESTE2_THREADS[$i]}
    FILE="$OUTPUT_DIR/out_teste2_${P}p_${T}t.txt"
    echo -e "${BLUE}• teste2 (${P}p, ${T}t)${NC}"
    compare_centroids "$FILE" "$SEQ_OUT"; highlight_check $? "Comparação com Sequencial"
    compare_centroids "$FILE" "$TESTE2_BASELINE_FILE"; highlight_check $? "Comparação com teste2 baseline (${TESTE2_BASELINE_P}p, ${TESTE2_BASELINE_T}t)"
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
for i in "${!WORKS1_PROCESSES[@]}"; do
    P=${WORKS1_PROCESSES[$i]}
    T=${WORKS1_THREADS[$i]}
    printf "%-36s %s\n" "MPI+OMP works1 (${P}p, ${T}t)" "${TIMES[mpiworks1_${P}p_${T}t]}"
done
for i in "${!TESTE2_PROCESSES[@]}"; do
    P=${TESTE2_PROCESSES[$i]}
    T=${TESTE2_THREADS[$i]}
    printf "%-36s %s\n" "teste2 (${P}p, ${T}t)" "${TIMES[teste2_${P}p_${T}t]}"
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
for i in "${!WORKS1_PROCESSES[@]}"; do
    P=${WORKS1_PROCESSES[$i]}
    T=${WORKS1_THREADS[$i]}
    sp=$(calc_speedup "$SEQ_T" "${TIMES[mpiworks1_${P}p_${T}t]}")
    printf "%-36s %s\n" "MPI+OMP works1 (${P}p, ${T}t)" "${sp}x"
    if [ "$sp" != "-" ] && echo "$sp > $best_speed" | bc -l >/dev/null 2>&1; then best_speed=$sp; best_label="MPI+OMP works1 (${P}p, ${T}t)"; fi
done
for i in "${!TESTE2_PROCESSES[@]}"; do
    P=${TESTE2_PROCESSES[$i]}
    T=${TESTE2_THREADS[$i]}
    sp=$(calc_speedup "$SEQ_T" "${TIMES[teste2_${P}p_${T}t]}")
    printf "%-36s %s\n" "teste2 (${P}p, ${T}t)" "${sp}x"
    if [ "$sp" != "-" ] && echo "$sp > $best_speed" | bc -l >/dev/null 2>&1; then best_speed=$sp; best_label="teste2 (${P}p, ${T}t)"; fi
done

echo -e "\n${BLUE}🏆 Melhor resultado:${NC} ${GREEN}${best_speed}x${NC} com ${YELLOW}${best_label}${NC}"

echo -e "\n${GREEN}Saídas salvas em:${NC} $OUTPUT_DIR"
