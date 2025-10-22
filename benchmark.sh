#!/bin/bash

# Script de Benchmark e Comparação K-Means
# Uso: ./benchmark.sh [arquivo_entrada]

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Arquivo de entrada (padrão: medium_30k_input.txt)
INPUT_FILE=${1:-UCI_Credit_Card.txt}

# Verifica se o arquivo de entrada existe
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}❌ Erro: Arquivo $INPUT_FILE não encontrado!${NC}"
    exit 1
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      BENCHMARK K-MEANS - SEQUENCIAL vs PARALELO        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}📁 Arquivo de entrada:${NC} $INPUT_FILE"
echo ""

# Compila os programas se necessário
echo -e "${YELLOW}🔨 Verificando compilação...${NC}"
if [ ! -f "kmeans_sequencial" ] || [ "kmeans_sequencial.cpp" -nt "kmeans_sequencial" ]; then
    echo "   Compilando versão sequencial..."
    g++ -std=c++17 -o kmeans_sequencial kmeans_sequencial.cpp -lm
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Erro na compilação do sequencial${NC}"
        exit 1
    fi
fi

if [ ! -f "kmeans_openmp" ] || [ "kmeans_openmp.cpp" -nt "kmeans_openmp" ]; then
    echo "   Compilando versão paralela..."
    g++ -std=c++17 -fopenmp -o kmeans_openmp kmeans_openmp.cpp -lm
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Erro na compilação do OpenMP${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}   ✅ Compilação OK${NC}"
echo ""

# Executa versão sequencial
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}⏱️  Executando SEQUENCIAL...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
TIME_SEQ=$( { time ./kmeans_sequencial output_seq.txt < $INPUT_FILE 2>&1; } 2>&1 | grep real | awk '{print $2}')
echo -e "${GREEN}✅ Concluído em: $TIME_SEQ${NC}"
echo ""

# Executa versão paralela com diferentes números de threads
THREADS_LIST=(1 2 4 8)
declare -A TIME_PAR

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}⏱️  Executando PARALELO (OpenMP)...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

for threads in "${THREADS_LIST[@]}"; do
    echo -e "${YELLOW}   🔧 Com $threads thread(s)...${NC}"
    TIME_PAR[$threads]=$( { time OMP_NUM_THREADS=$threads ./kmeans_openmp output_omp_${threads}t.txt < $INPUT_FILE 2>&1; } 2>&1 | grep real | awk '{print $2}')
    echo -e "${GREEN}   ✅ Concluído em: ${TIME_PAR[$threads]}${NC}"
done
echo ""

# Compara os resultados (centróides)
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}🔍 Verificando corretude dos resultados...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

grep "Cluster values:" output_seq.txt | sort > centroids_seq_bench.txt

ALL_CORRECT=true
for threads in "${THREADS_LIST[@]}"; do
    grep "Cluster values:" output_omp_${threads}t.txt | sort > centroids_omp_${threads}t_bench.txt
    
    if diff -q centroids_seq_bench.txt centroids_omp_${threads}t_bench.txt > /dev/null 2>&1; then
        echo -e "${GREEN}   ✅ OpenMP ($threads threads): Centróides IDÊNTICOS${NC}"
    else
        echo -e "${RED}   ❌ OpenMP ($threads threads): DIFERENÇA encontrada!${NC}"
        ALL_CORRECT=false
    fi
done
echo ""

# Calcula speedup
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  RESUMO DE PERFORMANCE                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Converte tempo para segundos para cálculo
time_to_seconds() {
    local time_str=$1
    # Remove 'm' e 's', substitui vírgula por ponto e converte para segundos
    local minutes=$(echo $time_str | sed 's/m.*//')
    local seconds=$(echo $time_str | sed 's/.*m//;s/s//;s/,/./')
    echo "$minutes * 60 + $seconds" | bc -l
}

SEQ_SECONDS=$(time_to_seconds $TIME_SEQ)

printf "${YELLOW}%-20s${NC} ${BLUE}%-12s${NC} ${GREEN}%-12s${NC}\n" "Configuração" "Tempo" "Speedup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "%-20s %-12s ${GREEN}%-12s${NC}\n" "Sequencial" "$TIME_SEQ" "1.00x"

for threads in "${THREADS_LIST[@]}"; do
    PAR_SECONDS=$(time_to_seconds ${TIME_PAR[$threads]})
    SPEEDUP=$(echo "scale=2; $SEQ_SECONDS / $PAR_SECONDS" | bc -l)
    printf "%-20s %-12s ${GREEN}%-12s${NC}\n" "OpenMP ($threads threads)" "${TIME_PAR[$threads]}" "${SPEEDUP}x"
done
echo ""

# Resultado final
if [ "$ALL_CORRECT" = true ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✅ TODOS OS RESULTADOS ESTÃO CORRETOS E IDÊNTICOS!    ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
else
    echo -e "${RED}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ❌ ATENÇÃO: DIFERENÇAS ENCONTRADAS NOS RESULTADOS!    ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════╝${NC}"
fi
echo ""

# Melhor speedup
BEST_THREADS=1
BEST_SPEEDUP=1.00
for threads in "${THREADS_LIST[@]}"; do
    PAR_SECONDS=$(time_to_seconds ${TIME_PAR[$threads]})
    SPEEDUP=$(echo "scale=2; $SEQ_SECONDS / $PAR_SECONDS" | bc -l)
    BETTER=$(echo "$SPEEDUP > $BEST_SPEEDUP" | bc -l)
    if [ "$BETTER" -eq 1 ]; then
        BEST_SPEEDUP=$SPEEDUP
        BEST_THREADS=$threads
    fi
done

echo -e "${BLUE}🏆 Melhor resultado: ${GREEN}${BEST_SPEEDUP}x de speedup com $BEST_THREADS thread(s)${NC}"
echo ""

# Limpeza opcional
read -p "Deseja remover os arquivos de saída? (s/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    rm -f output_seq.txt output_omp_*t.txt centroids_*_bench.txt
    echo -e "${GREEN}✅ Arquivos de saída removidos${NC}"
fi
