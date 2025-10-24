@echo off
rem Versao simplificada: mede em segundos (inteiros) usando %TIME%
setlocal enabledelayedexpansion

rem compilar os programas
g++ -o kmeans_sequencial kmeans_sequencial.cpp
g++ -o kmeans_openmp kmeans_openmp.cpp -fopenmp
g++ -o kmeans_mpi_openmp kmeans_mpi_openmp.cpp -I%MSMPI_INC% -L%MSMPI_LIB64% -lmsmpi -fopenmp

rem criar diretorio para resultados e variaveis de entrada/saida
set runsDir=runs
if not exist %runsDir% (
    mkdir %runsDir%
) else (
    rd /s /q %runsDir%
    mkdir %runsDir%
)
set timeOutputFile=%runsDir%\tempo.txt
set dataInput=UCI_Credit_Card.txt

rem medir tempo de execucao sequencial
echo Teste sequencial iniciado.
set start=%TIME%
.\kmeans_sequencial < %dataInput% > %runsDir%\seq
set end=%TIME%
call :calculo_tempo
set sequencial_elapsed=%elapsed%
call :print_execution_time sequencial




rem medir tempo de execucao openmp
echo Teste openmp iniciado.

set threads=(1 2 4 8)
for %%t in %threads% do (
    echo Teste openmp com %%t threads iniciado.
    set OMP_NUM_THREADS=%%t
    set "start=!TIME!"
    .\kmeans_openmp < %dataInput% > %runsDir%\openmp_%%t
    set "end=!TIME!"
    call :calculo_tempo
    call :print_execution_time openmp_%%t
)

rem Exemplo: executa (1p,1t), (1p,2t), (1p,4t), (2p,2t), (4p,1t)

for %%p in ("1,1" "1,2" "1,4" "2,2" "4,1") do (
    for /f "tokens=1,2 delims=," %%a in ("%%~p") do (
        echo Teste openmp+mpi com %%a processos e %%b threads iniciado.
        set OMP_NUM_THREADS=%%b
        set "start=!TIME!"
        mpiexec -n %%a .\kmeans_mpi_openmp < %dataInput% > %runsDir%\openmp_mpi_openmp_%%a_procs_%%b_threads
        set "end=!TIME!"
        call :calculo_tempo
        call :print_execution_time openmp_mpi_openmp_%%a_procs_%%b_threads
    )
)

exit /b 0

:print_execution_time
rem determina status default
set "status=sucesso"
rem calcula speedup usando PowerShell (2 casas decimais)
set "speedup=NA"
if defined sequencial_elapsed (
    for /f "delims=" %%A in ('powershell -NoProfile -Command "$cur=[double]('!elapsed!'); $seq=[double]('!sequencial_elapsed!'); if($cur -eq 0){'inf'} else {('{0:F2}' -f ($seq/$cur)) }"') do set "speedup=%%A"
)

rem se for o teste sequencial, não compara
if /I "%~1"=="sequencial" (
    set "status=sequencial"
) else (
    rem verifica existência dos arquivos
    if not exist "%runsDir%\seq" (
        set "status=erro"
    ) else if not exist "%runsDir%\%~1" (
        set "status=erro"
    ) else (
        fc "%runsDir%\seq" "%runsDir%\%~1" > nul
        if errorlevel 1 (
            set "status=erro"
        ) else (
            set "status=sucesso"
        )
    )
)

rem imprime sempre uma linha clara no console
if "%status%"=="sequencial" (
    echo Teste %~1 finalizado ^(sequencial^). Tempo: !elapsed! s speedup: !speedup! 
    echo Execution time %~1: !elapsed! s  speedup: !speedup!  >> %timeOutputFile%
) else if "%status%"=="sucesso" (
    echo Teste %~1 finalizado com sucesso. Tempo: !elapsed! s speedup: !speedup!
    echo Execution time %~1: !elapsed! s  speedup: !speedup!  >> %timeOutputFile%
) else (
    echo Teste %~1 finalizado com erro. Tempo: !elapsed! s speedup: !speedup!
    echo Execution time error %~1: !elapsed! s  speedup: !speedup!  >> %timeOutputFile%
)

exit /b 0






:calculo_tempo
rem calcula diferença de tempo usando PowerShell (segundos com 2 casas decimais)
for /f "delims=" %%A in ('powershell -NoProfile -Command "$s='!start!'.Replace(',','.'); $e='!end!'.Replace(',','.'); $st=[datetime]::Parse($s); $et=[datetime]::Parse($e); if($et -lt $st){$et=$et.AddDays(1)}; '{0:F2}' -f ($et - $st).TotalSeconds"') do set "elapsed=%%A"
exit /b 0