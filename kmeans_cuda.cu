#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <cstring>

using namespace std;

// =============================================================
// PARTE 1: KERNELS OTIMIZADOS
// =============================================================

// Kernel 1: Atribui labels
__global__ void assign_labels_kernel(const double* points, const double* centroids, 
	int* labels, const int* old_labels, int N, int D, int K, int* changes_out) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	// Ponteiro para o ponto atual
	const double* p = points + (size_t)idx * D;
	int best = 0;
	double min_dist_sq = 0.0;
	// Calcula distância inicial para o cluster 0
	for (int j = 0; j < D; j++) {
		double diff = centroids[0 * D + j] - p[j];
		min_dist_sq += diff * diff;
	}
	// Verifica os outros clusters
	for (int c = 1; c < K; c++) {
		double dist_sq = 0.0;
		for (int j = 0; j < D; j++) {
			double diff = centroids[c * D + j] - p[j];
			dist_sq += diff * diff;
		}
		if (dist_sq < min_dist_sq) {
			min_dist_sq = dist_sq;
			best = c;
		}
	}
	labels[idx] = best;
	// Verifica se houve mudança de cluster
	if (old_labels != nullptr) {
		int oldv = old_labels[idx];
		if (oldv != best) {
			atomicAdd(changes_out, 1);
		}
	} else {
		atomicAdd(changes_out, 1);
	}
}

// Kernel 2: Acumula somas e contagens usando Memória Compartilhada
__global__ void accumulate_kernel_shared(const double* points, const int* 
	labels, double* sums, int* counts, int N, int D, int K) {
	// 1. Declara memória compartilhada dinâmica
	// Layout: [Somas (double)] ... [Contagens (int)]
	extern __shared__ double s_mem[];
	double* s_sums = s_mem;
	int* s_counts = (int*)&s_sums[K * D];
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	// 2. Inicializa memória compartilhada com 0
	for (int i = tid; i < K * D; i += blockDim.x) {
		s_sums[i] = 0.0;
	}
	for (int i = tid; i < K; i += blockDim.x) {
		s_counts[i] = 0;
	}
	__syncthreads();
	// 3. Acumula na memória compartilhada
	if (gid < N) {
		int lbl = labels[gid];
		const double* p = points + (size_t)gid * D;
		// Soma atômica local
		atomicAdd(&s_counts[lbl], 1);
		for (int j = 0; j < D; j++) {
			atomicAdd(&s_sums[lbl * D + j], p[j]);
		}
	}
	__syncthreads();
	// 4. Transfere o resumo do bloco para a memória global
	for (int i = tid; i < K * D; i += blockDim.x) {
		if (s_sums[i] != 0.0) {
			atomicAdd(&sums[i], s_sums[i]);
		}
	}
	for (int i = tid; i < K; i += blockDim.x) {
		if (s_counts[i] != 0) {
			atomicAdd(&counts[i], s_counts[i]);
		}
	}
}

// =============================================================
// CLASSES AUXILIARES (Point e Cluster)
// =============================================================

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

	Point() : id_point(-1), total_values(0), id_cluster(-1) {}

	int getID() { return id_point; }
	void setCluster(int id_cluster) { this->id_cluster = id_cluster; }
	int getCluster() { return id_cluster; }
	double getValue(int index) { return values[index]; }
	int getTotalValues() { return total_values; }
	void addValue(double value) { values.push_back(value); }
	string getName() { return name; }
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

	Cluster(int id_cluster, vector<double>& central_vals)
	{
		this->id_cluster = id_cluster;
		central_values = central_vals;
	}

	Cluster() : id_cluster(-1) {}

	void addPoint(Point point) { points.push_back(point); }
	
	bool removePoint(int id_point)
	{
		int total_points = points.size();
		for(int i = 0; i < total_points; i++) {
			if(points[i].getID() == id_point) {
				points.erase(points.begin() + i);
				return true;
			}
		}
		return false;
	}

	double getCentralValue(int index) { return central_values[index]; }
	void setCentralValue(int index, double value) { central_values[index] = value; }
	Point getPoint(int index) { return points[index]; }
	int getTotalPoints() { return points.size(); }
	int getID() { return id_cluster; }
	void clearPoints() { points.clear(); }
};

// ========================================
// Classe KMeans
// ========================================
class KMeans
{
private:
	int K;
	int total_values, total_points, max_iterations;
	vector<Cluster> clusters;

public:
	KMeans(int K, int total_points, int total_values, int max_iterations)
	{
		this->K = K;
		this->total_points = total_points;
		this->total_values = total_values;
		this->max_iterations = max_iterations;
	}

	void run(vector<Point> & points, ostream& out, int gpu_threads = 256)
	{
		if(K > total_points) return;

		// --- Inicialização dos Clusters (Aleatória) ---
		{
			vector<int> prohibited_indexes;
			for(int i = 0; i < K; i++)
			{
				while(true)
				{
					int index_point = rand() % total_points;
					if(find(prohibited_indexes.begin(), prohibited_indexes.end(), index_point) == prohibited_indexes.end())
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

		// Prepara vetor plano de centroides
		vector<double> centroids_flat(K * total_values);
		for(int i = 0; i < K; i++)
			for(int j = 0; j < total_values; j++)
				centroids_flat[i * total_values + j] = clusters[i].getCentralValue(j);

		// Prepara dados para GPU
		int my_count = total_points;
		vector<double> local_points_flat(my_count * total_values);
		vector<int> host_labels(my_count);

		for (int i = 0; i < my_count; ++i) {
			for (int j = 0; j < total_values; ++j)
				local_points_flat[i * total_values + j] = points[i].getValue(j);
			host_labels[i] = points[i].getCluster();
		}

		// Alocação na GPU
		double *d_points, *d_centroids, *d_sums;
		int *d_labels, *d_old_labels, *d_changes, *d_counts;

		cudaMalloc((void**)&d_points, sizeof(double) * (size_t)my_count * total_values);
		cudaMemcpy(d_points, local_points_flat.data(), sizeof(double) * (size_t)my_count * total_values, cudaMemcpyHostToDevice);
		
		cudaMalloc((void**)&d_labels, sizeof(int) * my_count);
		cudaMemcpy(d_labels, host_labels.data(), sizeof(int) * my_count, cudaMemcpyHostToDevice);
		
		cudaMalloc((void**)&d_old_labels, sizeof(int) * my_count);
		cudaMalloc((void**)&d_changes, sizeof(int));

		cudaMalloc((void**)&d_centroids, sizeof(double) * (size_t)K * total_values);
		cudaMalloc((void**)&d_sums, sizeof(double) * (size_t)K * total_values);
		cudaMalloc((void**)&d_counts, sizeof(int) * K);

		// LOOP PRINCIPAL
		int iter = 1;
		while(true)
		{
			int local_changes = 0; 
			
			// --- Passo 1: Atribuir Labels ---
			cudaMemcpy(d_centroids, centroids_flat.data(), sizeof(double) * (size_t)K * total_values, cudaMemcpyHostToDevice);
			cudaMemcpy(d_old_labels, d_labels, sizeof(int) * my_count, cudaMemcpyDeviceToDevice);
			cudaMemset(d_changes, 0, sizeof(int));

			int threads = gpu_threads > 0 ? gpu_threads : 256;
			int blocks = (my_count + threads - 1) / threads;
			
			assign_labels_kernel<<<blocks, threads>>>(d_points, d_centroids, d_labels, d_old_labels, my_count, total_values, K, d_changes);
			cudaDeviceSynchronize();

			cudaMemcpy(&local_changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost);

			// --- Passo 2: Recalcular Centroides (Acumular) ---
			cudaMemset(d_sums, 0, sizeof(double) * (size_t)K * total_values);
			cudaMemset(d_counts, 0, sizeof(int) * K);

			// Calcula tamanho da memória compartilhada necessária
			size_t smem_size = (K * total_values * sizeof(double)) + (K * sizeof(int));
			
			accumulate_kernel_shared<<<blocks, threads, smem_size>>>(d_points, d_labels, d_sums, d_counts, my_count, total_values, K);
			cudaDeviceSynchronize();

			// Traz resultados de volta
			vector<double> global_sums(K * total_values);
			vector<int> global_counts(K);
			cudaMemcpy(global_sums.data(), d_sums, sizeof(double) * (size_t)K * total_values, cudaMemcpyDeviceToHost);
			cudaMemcpy(global_counts.data(), d_counts, sizeof(int) * K, cudaMemcpyDeviceToHost);

			// Atualiza centroides na CPU
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

			if(local_changes == 0 || iter >= max_iterations) break;
			iter++;
		}

		// --- Finalização ---
		// Atualiza labels finais
		cudaMemcpy(host_labels.data(), d_labels, sizeof(int) * my_count, cudaMemcpyDeviceToHost);
		for (int i = 0; i < my_count; ++i) points[i].setCluster(host_labels[i]);

		// Reconstrói clusters para output
		for(int i = 0; i < K; i++) clusters[i].clearPoints();
		for(int i = 0; i < total_points; i++) {
			int c_id = points[i].getCluster();
			if(c_id >= 0 && c_id < K) clusters[c_id].addPoint(points[i]);
		}

		// Imprime
		for(int i = 0; i < K; i++)
		{
			out << "Cluster values: ";
			for(int j = 0; j < total_values; j++) out << clusters[i].getCentralValue(j) << " ";
			out << "\n\n";
		}

		cudaFree(d_points); cudaFree(d_centroids); cudaFree(d_labels);
		cudaFree(d_old_labels); cudaFree(d_changes); cudaFree(d_sums); cudaFree(d_counts);
	}
};

int main(int argc, char *argv[])
{
	srand(42);

	int total_points, total_values, K, max_iterations, has_name;
	if (!(cin >> total_points >> total_values >> K >> max_iterations >> has_name)) return 0;

	vector<Point> points;
	points.reserve(total_points);
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

		if(has_name) {
			cin >> point_name;
			points.push_back(Point(i, values, point_name));
		} else {
			points.push_back(Point(i, values));
		}
	}

	KMeans kmeans(K, total_points, total_values, max_iterations);
	
	// Default threads agora pode ser maior (128 ou 256) graças à Shared Memory
	int gpu_threads = 256; 

	ofstream outfile;
	ostream* output = &cout;
	
	if(argc > 1) {
		outfile.open(argv[1]);
		if(outfile.is_open()) output = &outfile;
	}

	if(argc > 2) {
		try { gpu_threads = stoi(argv[2]); } catch(...) { gpu_threads = 256; }
	}
	
	kmeans.run(points, *output, gpu_threads);
	
	if(outfile.is_open()) outfile.close();

	return 0;
}