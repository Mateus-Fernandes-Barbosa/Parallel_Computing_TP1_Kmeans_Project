// Implementation of the KMeans Algorithm with MPI + OpenMP
// Mantém a estrutura original do código sequencial/OpenMP
// Lógica: Rank 0 (pai) coordena, ranks filhos processam suas porções

// how to compile:
// mpicxx -std=c++17 -O0 -fopenmp -o kmeans_mpi_simple kmeans_mpi_simple.cpp -lm

// how to run:
// OMP_NUM_THREADS=2 mpirun -np 4 ./kmeans_mpi_simple output.txt < input.txt

#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <omp.h>
#include <mpi.h>

using namespace std;

// ========================================
// Classes originais mantidas (Point, Cluster)
// ========================================
class Point
{
private:
	int id_point, id_cluster;
	vector<double> values;
	int total_values;
	string name;

public:
	Point(int id_point, vector<double>& values, string name = "")
	{
		this->id_point = id_point;
		total_values = values.size();

		for(int i = 0; i < total_values; i++)
			this->values.push_back(values[i]);

		this->name = name;
		id_cluster = -1;
	}

	// Construtor vazio para facilitar arrays
	Point() : id_point(-1), total_values(0), id_cluster(-1) {}

	int getID()
	{
		return id_point;
	}

	void setCluster(int id_cluster)
	{
		this->id_cluster = id_cluster;
	}

	int getCluster()
	{
		return id_cluster;
	}

	double getValue(int index)
	{
		return values[index];
	}

	int getTotalValues()
	{
		return total_values;
	}

	void addValue(double value)
	{
		values.push_back(value);
	}

	string getName()
	{
		return name;
	}

	// Métodos auxiliares para MPI
	void setID(int id) { id_point = id; }
	void setTotalValues(int tv) { total_values = tv; values.resize(tv); }
	void setValue(int index, double val) { 
		if(index >= 0 && index < (int)values.size()) 
			values[index] = val; 
	}
	void setName(string n) { name = n; }
};

class Cluster
{
private:
	int id_cluster;
	vector<double> central_values;
	vector<Point> points;

public:
	Cluster(int id_cluster, Point point)
	{
		this->id_cluster = id_cluster;

		int total_values = point.getTotalValues();

		for(int i = 0; i < total_values; i++)
			central_values.push_back(point.getValue(i));

		points.push_back(point);
	}

	// Construtor alternativo para inicializar apenas com centroides
	Cluster(int id_cluster, vector<double>& central_vals)
	{
		this->id_cluster = id_cluster;
		central_values = central_vals;
	}

	Cluster() : id_cluster(-1) {}

	void addPoint(Point point)
	{
		points.push_back(point);
	}

	bool removePoint(int id_point)
	{
		int total_points = points.size();

		for(int i = 0; i < total_points; i++)
		{
			if(points[i].getID() == id_point)
			{
				points.erase(points.begin() + i);
				return true;
			}
		}
		return false;
	}

	double getCentralValue(int index)
	{
		return central_values[index];
	}

	void setCentralValue(int index, double value)
	{
		central_values[index] = value;
	}

	Point getPoint(int index)
	{
		return points[index];
	}

	int getTotalPoints()
	{
		return points.size();
	}

	int getID()
	{
		return id_cluster;
	}

	void clearPoints()
	{
		points.clear();
	}
};

// ========================================
// Classe KMeans adaptada para MPI
// ========================================
class KMeans
{
private:
	int K; // número de clusters
	int total_values, total_points, max_iterations;
	vector<Cluster> clusters;
	int rank, num_procs; // MPI rank e número de processos

	// Retorna ID do cluster mais próximo (distância euclidiana)
	// MANTIDO IGUAL AO ORIGINAL
	int getIDNearestCenter(Point point)
	{
		double sum = 0.0, min_dist;
		int id_cluster_center = 0;

		for(int i = 0; i < total_values; i++)
		{
			sum += pow(clusters[0].getCentralValue(i) -
					   point.getValue(i), 2.0);
		}

		min_dist = sqrt(sum);

		for(int i = 1; i < K; i++)
		{
			double dist;
			sum = 0.0;

			for(int j = 0; j < total_values; j++)
			{
				sum += pow(clusters[i].getCentralValue(j) -
						   point.getValue(j), 2.0);
			}

			dist = sqrt(sum);

			if(dist < min_dist)
			{
				min_dist = dist;
				id_cluster_center = i;
			}
		}

		return id_cluster_center;
	}

public:
	KMeans(int K, int total_points, int total_values, int max_iterations, int rank, int num_procs)
	{
		this->K = K;
		this->total_points = total_points;
		this->total_values = total_values;
		this->max_iterations = max_iterations;
		this->rank = rank;
		this->num_procs = num_procs;
	}

	void run(vector<Point> & points, ostream& out)
	{
		if(K > total_points)
			return;

		// ========================================
		// ETAPA 1: Rank 0 inicializa clusters
		// ========================================
		if(rank == 0)
		{
			vector<int> prohibited_indexes;

			// Escolhe K pontos distintos como centroides iniciais
			// MANTIDO IGUAL AO ORIGINAL
			for(int i = 0; i < K; i++)
			{
				while(true)
				{
					int index_point = rand() % total_points;

					if(find(prohibited_indexes.begin(), prohibited_indexes.end(),
							index_point) == prohibited_indexes.end())
					{
						prohibited_indexes.push_back(index_point);
						points[index_point].setCluster(i);
						Cluster cluster(i, points[index_point]);
						clusters.push_back(cluster);
						break;
					}
				}
			}
		}
		else
		{
			// Processos filhos inicializam clusters vazios
			for(int i = 0; i < K; i++)
			{
				vector<double> temp_central(total_values, 0.0);
				Cluster cluster(i, temp_central);
				clusters.push_back(cluster);
			}
		}

		// ========================================
		// ETAPA 2: Broadcast dos centroides iniciais
		// ========================================
		vector<double> centroids_flat(K * total_values);
		
		if(rank == 0)
		{
			// Empacota centroides em array flat
			for(int i = 0; i < K; i++)
				for(int j = 0; j < total_values; j++)
					centroids_flat[i * total_values + j] = clusters[i].getCentralValue(j);
		}

		// Todos os processos recebem os centroides
		MPI_Bcast(centroids_flat.data(), K * total_values, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// Processos filhos atualizam seus clusters com centroides recebidos
		if(rank != 0)
		{
			for(int i = 0; i < K; i++)
				for(int j = 0; j < total_values; j++)
					clusters[i].setCentralValue(j, centroids_flat[i * total_values + j]);
		}

		// ========================================
		// Divisão dos pontos entre processos
		// ========================================
		int points_per_proc = total_points / num_procs;
		int remainder = total_points % num_procs;
		
		// Cada processo calcula seu início e quantidade de pontos
		int my_start = rank * points_per_proc + min(rank, remainder);
		int my_count = points_per_proc + (rank < remainder ? 1 : 0);

		// ========================================
		// LOOP PRINCIPAL (iterações do K-means)
		// ========================================
		int iter = 1;

		while(true)
		{
			int local_changes = 0; // Contador local de mudanças

			// ========================================
			// ETAPA 4: Atribuir pontos aos clusters mais próximos
			// (Cada processo trabalha em sua porção)
			// ========================================
			#pragma omp parallel for schedule(static) reduction(+:local_changes)
			for(int i = my_start; i < my_start + my_count; i++)
			{
				int id_old_cluster = points[i].getCluster();
				int id_nearest_center = getIDNearestCenter(points[i]);

        
				if(id_old_cluster != id_nearest_center)
				{
          #pragma omp critical
					{
						if(id_old_cluster != -1)
							clusters[id_old_cluster].removePoint(points[i].getID());
							points[i].setCluster(id_nearest_center);
							clusters[id_nearest_center].addPoint(points[i]);
					}
					local_changes++;
				}
			}

			// ========================================
			// ETAPA 5: Somar mudanças de todos os processos
			// ========================================
			int global_changes = 0;
			MPI_Allreduce(&local_changes, &global_changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

			// ========================================
			// Recalcular centroides
			// ========================================
			vector<double> local_sums(K * total_values, 0.0);
			vector<int> local_counts(K, 0);

			#pragma omp parallel
			{
				vector<double> thread_sums(K * total_values, 0.0);
				vector<int> thread_counts(K, 0);

				#pragma omp for schedule(static)
				for(int i = my_start; i < my_start + my_count; i++)
				{
					int cluster_id = points[i].getCluster();
					if(cluster_id >= 0 && cluster_id < K)
					{
						thread_counts[cluster_id]++;
						for(int j = 0; j < total_values; j++)
						{
							thread_sums[cluster_id * total_values + j] += points[i].getValue(j);
						}
					}
				}

				// Redução das threads
				#pragma omp critical
				{
					for(int c = 0; c < K; c++)
					{
						local_counts[c] += thread_counts[c];
						for(int j = 0; j < total_values; j++)
							local_sums[c * total_values + j] += thread_sums[c * total_values + j];
					}
				}
			}

			// ========================================
			// Agregar somas de todos os processos
			// ========================================
			vector<double> global_sums(K * total_values, 0.0);
			vector<int> global_counts(K, 0);

			MPI_Allreduce(local_sums.data(), global_sums.data(), 
						  K * total_values, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(local_counts.data(), global_counts.data(), 
						  K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

			// ========================================
			// Rank 0 atualiza centroides
			// ========================================
			if(rank == 0)
			{
        #pragma omp parallel for schedule(static)
				for(int i = 0; i < K; i++)
				{
					if(global_counts[i] > 0)
					{
						for(int j = 0; j < total_values; j++)
						{
							double new_val = global_sums[i * total_values + j] / global_counts[i];
							clusters[i].setCentralValue(j, new_val);
							centroids_flat[i * total_values + j] = new_val;
						}
					}
				}
			}

			// ========================================
			// Broadcast dos novos centroides
			// ========================================
			MPI_Bcast(centroids_flat.data(), K * total_values, MPI_DOUBLE, 0, MPI_COMM_WORLD);

			// Processos filhos atualizam seus clusters
			if(rank != 0)
			{
				for(int i = 0; i < K; i++)
					for(int j = 0; j < total_values; j++)
						clusters[i].setCentralValue(j, centroids_flat[i * total_values + j]);
			}

			// ========================================
			// ETAPA 10: Verificar convergência
			// ========================================
			if(global_changes == 0 || iter >= max_iterations)
			{
				if(rank == 0)
					out << "Break in iteration " << iter << "\n\n";
				break;
			}

			iter++;
		}

		// ========================================
		// ETAPA 11: Sincronizar cluster IDs de todos os processos
		// ========================================
		
		// Cada processo prepara seus cluster_ids locais
		vector<int> local_cluster_ids(my_count);
		for(int i = 0; i < my_count; i++)
		{
			local_cluster_ids[i] = points[my_start + i].getCluster();
		}
		
		// Preparar arrays de contagens e deslocamentos para MPI_Gatherv
		vector<int> recv_counts(num_procs);
		vector<int> recv_displs(num_procs);
		
		for(int r = 0; r < num_procs; r++)
		{
			int r_count = points_per_proc + (r < remainder ? 1 : 0);
			int r_start = r * points_per_proc + min(r, remainder);
			recv_counts[r] = r_count;
			recv_displs[r] = r_start;
		}
		
		// Vetor para receber todos os cluster_ids (apenas rank 0 precisa alocar)
		vector<int> all_cluster_ids;
		if(rank == 0)
		{
			all_cluster_ids.resize(total_points);
		}
		
		// Rank 0 coleta todos os cluster_ids de todos os processos
		MPI_Gatherv(local_cluster_ids.data(), my_count, MPI_INT,
					all_cluster_ids.data(), recv_counts.data(), recv_displs.data(), MPI_INT,
					0, MPI_COMM_WORLD);
		
		// ========================================
		// ETAPA 12: Rank 0 imprime resultados
		// ========================================
		if(rank == 0)
		{
			// Atualizar TODOS os pontos com os cluster_ids recebidos
			for(int i = 0; i < total_points; i++)
			{
				points[i].setCluster(all_cluster_ids[i]);
			}
			
			// Reconstrói clusters para impressão
			for(int i = 0; i < K; i++)
				clusters[i].clearPoints();

			for(int i = 0; i < total_points; i++)
			{
				int cluster_id = points[i].getCluster();
				if(cluster_id >= 0 && cluster_id < K)
					clusters[cluster_id].addPoint(points[i]);
			}

			// Imprime resultados (MANTIDO IGUAL AO ORIGINAL)
			for(int i = 0; i < K; i++)
			{
				int total_points_cluster = clusters[i].getTotalPoints();

				out << "Cluster " << clusters[i].getID() + 1 << endl;
				for(int j = 0; j < total_points_cluster; j++)
				{
					out << "Point " << clusters[i].getPoint(j).getID() + 1 << ": ";
					for(int p = 0; p < total_values; p++)
						out << clusters[i].getPoint(j).getValue(p) << " ";

					string point_name = clusters[i].getPoint(j).getName();

					if(point_name != "")
						out << "- " << point_name;

					out << endl;
				}

				out << "Cluster values: ";

				for(int j = 0; j < total_values; j++)
					out << clusters[i].getCentralValue(j) << " ";

				out << "\n\n";
			}
		}

		// Processos filhos terminam aqui (inativados)
	}
};

int main(int argc, char *argv[])
{
	// ========================================
	// Inicialização MPI
	// ========================================
	MPI_Init(&argc, &argv);

	int rank, num_procs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	// Seed fixa para resultados reproduzíveis
	srand(42);

	int total_points, total_values, K, max_iterations, has_name;

	// ========================================
	// Rank 0 lê entrada
	// ========================================
	if(rank == 0)
	{
		cin >> total_points >> total_values >> K >> max_iterations >> has_name;
	}

	// Broadcast dos parâmetros para todos os processos
	MPI_Bcast(&total_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&total_values, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&max_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&has_name, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// ========================================
	// Preparar array de pontos (todos os processos)
	// ========================================
	vector<Point> points;
	points.reserve(total_points);

	if(rank == 0)
	{
		// Rank 0 lê os dados
		string point_name;
		for(int i = 0; i < total_points; i++)
		{
			vector<double> values;
			for(int j = 0; j < total_values; j++)
			{
				double value;
				cin >> value;
				values.push_back(value);
			}

			if(has_name)
			{
				cin >> point_name;
				Point p(i, values, point_name);
				points.push_back(p);
			}
			else
			{
				Point p(i, values);
				points.push_back(p);
			}
		}
	}
	else
	{
		// Processos filhos criam estrutura vazia
		for(int i = 0; i < total_points; i++)
		{
			vector<double> values(total_values, 0.0);
			Point p(i, values);
			points.push_back(p);
		}
	}

	// ========================================
	// Broadcast de todos os pontos para todos os processos
	// ========================================
	vector<double> all_points_flat(total_points * total_values);
	
	if(rank == 0)
	{
		// Empacota pontos em array flat
		for(int i = 0; i < total_points; i++)
			for(int j = 0; j < total_values; j++)
				all_points_flat[i * total_values + j] = points[i].getValue(j);
	}

	// Broadcast dos valores dos pontos
	MPI_Bcast(all_points_flat.data(), total_points * total_values, 
			  MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Processos filhos desempacotam
	if(rank != 0)
	{
		for(int i = 0; i < total_points; i++)
			for(int j = 0; j < total_values; j++)
				points[i].setValue(j, all_points_flat[i * total_values + j]);
	}

	// ========================================
	// Executar K-means
	// ========================================
	KMeans kmeans(K, total_points, total_values, max_iterations, rank, num_procs);
	
	// Apenas rank 0 escreve saída
	ofstream outfile;
	ostream* output = &cout;
	
	if(rank == 0 && argc > 1)
	{
		outfile.open(argv[1]);
		if(outfile.is_open())
			output = &outfile;
	}
	
	// Todos executam, mas apenas rank 0 imprime
	if(rank == 0)
		kmeans.run(points, *output);
	else
		kmeans.run(points, cout); // Filhos usam cout (ignorado)
	
	if(rank == 0 && outfile.is_open())
		outfile.close();

	// ========================================
	// Finalização MPI
	// ========================================
	MPI_Finalize();

	return 0;
}
