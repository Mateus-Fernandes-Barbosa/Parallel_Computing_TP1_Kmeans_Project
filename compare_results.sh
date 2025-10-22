#!/bin/bash

# Script simples para comparar apenas os centróides
# Uso: ./compare_results.sh output1.txt output2.txt

if [ $# -ne 2 ]; then
    echo "Uso: $0 arquivo1.txt arquivo2.txt"
    echo "Exemplo: $0 output_seq.txt output_omp.txt"
    exit 1
fi

FILE1=$1
FILE2=$2

if [ ! -f "$FILE1" ]; then
    echo "❌ Erro: $FILE1 não encontrado!"
    exit 1
fi

if [ ! -f "$FILE2" ]; then
    echo "❌ Erro: $FILE2 não encontrado!"
    exit 1
fi

echo "🔍 Comparando resultados..."
echo "  Arquivo 1: $FILE1"
echo "  Arquivo 2: $FILE2"
echo ""

# Extrai e ordena os centróides
grep "Cluster values:" "$FILE1" | sort > /tmp/centroids1.txt
grep "Cluster values:" "$FILE2" | sort > /tmp/centroids2.txt

# Conta quantos clusters foram encontrados
N_CLUSTERS1=$(wc -l < /tmp/centroids1.txt)
N_CLUSTERS2=$(wc -l < /tmp/centroids2.txt)

echo "📊 Clusters encontrados:"
echo "  $FILE1: $N_CLUSTERS1 clusters"
echo "  $FILE2: $N_CLUSTERS2 clusters"
echo ""

if [ "$N_CLUSTERS1" -ne "$N_CLUSTERS2" ]; then
    echo "❌ DIFERENÇA: Número de clusters diferente!"
    exit 1
fi

# Compara os centróides
if diff -q /tmp/centroids1.txt /tmp/centroids2.txt > /dev/null 2>&1; then
    echo "✅ RESULTADOS IDÊNTICOS!"
    echo "   Todos os $N_CLUSTERS1 centróides são iguais."
    
    # Mostra alguns exemplos
    echo ""
    echo "📋 Primeiros 3 centróides:"
    head -3 /tmp/centroids1.txt | nl -v 1 -w 2 -s '. '
    
    exit 0
else
    echo "❌ DIFERENÇA ENCONTRADA!"
    echo ""
    echo "Diferenças nos centróides:"
    diff --side-by-side --width=120 /tmp/centroids1.txt /tmp/centroids2.txt | head -20
    exit 1
fi
