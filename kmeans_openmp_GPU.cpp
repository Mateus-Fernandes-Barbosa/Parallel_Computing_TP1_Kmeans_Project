// Implementation of the KMeans Algorithm
// reference: http://mnemstudio.org/clustering-k-means-example-1.htm

// how to run:
// g++ -std=c++17 -fopenmp -o kmeans_openmp kmeans_openmp.cpp -lm
// ./kmeans_openmp < input.txt > output_openmp.txt

#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <omp.h>

using namespace std;

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

	// Remove todos os pontos do cluster
	void clearPoints()
	{
		points.clear();
	}
};

class KMeans
{
private:
	int K; // number of clusters
	int total_values, total_points, max_iterations;
	vector<Cluster> clusters;

	// return ID of nearest center (uses euclidean distance)
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
	KMeans(int K, int total_points, int total_values, int max_iterations)
	{
		this->K = K;
		this->total_points = total_points;
		this->total_values = total_values;
		this->max_iterations = max_iterations;
	}

	void run(vector<Point> & points, ostream& out)
	{
		if(K > total_points)
			return;

		vector<int> prohibited_indexes;

		// choose K distinct values for the centers of the clusters
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

		int iter = 1;

		while(true)
		{
			bool done = true;

			// Versão com offload para dispositivo (GPU) usando OpenMP target.
			// Estratégia:
			// 1) Extrair dados para arrays planos (pontos e centros) para tornar os dados offloadable.
			// 2) Offload: calcular, para cada ponto, o id do centro mais próximo (new_clusters).
			// 3) Atualizar no host se houve mudança (done) e reconstruir os vetores de pontos dos clusters.
			// 4) Offload: acumular somas por cluster (sums) e contagens (counts) usando atomic no dispositivo.
			// 5) Atualizar centros no host com sums / counts.

			// Aloca arrays planos
			double *points_values = new double[total_points * total_values];
			int *old_clusters = new int[total_points];
			int *new_clusters = new int[total_points];
			double *centers = new double[K * total_values];

			for(int i = 0; i < total_points; i++)
			{
				old_clusters[i] = points[i].getCluster();
				for(int j = 0; j < total_values; j++)
					points_values[i * total_values + j] = points[i].getValue(j);
			}

			for(int c = 0; c < K; c++)
			{
				for(int j = 0; j < total_values; j++)
					centers[c * total_values + j] = clusters[c].getCentralValue(j);
			}

			// Offload: calcula o centro mais próximo para cada ponto
			#pragma omp target teams distribute parallel for map(to: points_values[0:total_points*total_values], centers[0:K*total_values]) map(from: new_clusters[0:total_points])
			for(int i = 0; i < total_points; i++)
			{
				int best = 0;
				double min_sum = 0.0;
				// calcula a distância para o centro 0
				for(int v = 0; v < total_values; v++)
				{
					double diff = centers[0 * total_values + v] - points_values[i * total_values + v];
					min_sum += diff * diff;
				}
				for(int c = 1; c < K; c++)
				{
					double sum = 0.0;
					for(int v = 0; v < total_values; v++)
					{
						double diff = centers[c * total_values + v] - points_values[i * total_values + v];
						sum += diff * diff;
					}
					if(sum < min_sum)
					{
						min_sum = sum;
						best = c;
					}
				}
				new_clusters[i] = best;
			}

			// Atualiza done e os clusters no host
			done = true;
			for(int i = 0; i < total_points; i++)
			{
				if(old_clusters[i] != new_clusters[i])
					done = false;
				points[i].setCluster(new_clusters[i]);
			}

			// Reconstrói os vetores de pontos dos clusters (clear + add)
			for(int c = 0; c < K; c++)
				clusters[c].clearPoints();

			for(int i = 0; i < total_points; i++)
			{
				clusters[new_clusters[i]].addPoint(points[i]);
			}

			// Offload: acumula somas por cluster para recalcular centros
			double *sums = new double[K * total_values];
			int *counts = new int[K];
			// inicializa
			for(int i = 0; i < K * total_values; i++) sums[i] = 0.0;
			for(int i = 0; i < K; i++) counts[i] = 0;

			#pragma omp target teams distribute parallel for map(to: points_values[0:total_points*total_values], new_clusters[0:total_points]) map(tofrom: sums[0:K*total_values], counts[0:K])
			for(int i = 0; i < total_points; i++)
			{
				int c = new_clusters[i];
				#pragma omp atomic
				counts[c]++;
				for(int v = 0; v < total_values; v++)
				{
					#pragma omp atomic
					sums[c * total_values + v] += points_values[i * total_values + v];
				}
			}

			// Atualiza centros no host
			for(int c = 0; c < K; c++)
			{
				int cnt = counts[c];
				if(cnt > 0)
				{
					for(int v = 0; v < total_values; v++)
					{
						double newc = sums[c * total_values + v] / cnt;
						clusters[c].setCentralValue(v, newc);
					}
				}
			}

			// libera arrays
			delete[] points_values;
			delete[] old_clusters;
			delete[] new_clusters;
			delete[] centers;
			delete[] sums;
			delete[] counts;

			if(done == true || iter >= max_iterations)
			{
				// out << "Break in iteration " << iter << "\n\n";
				break;
			}

			iter++;
		}

		// shows elements of clusters
		for(int i = 0; i < K; i++)
		{
			int total_points_cluster =  clusters[i].getTotalPoints();

			// out << "Cluster " << clusters[i].getID() + 1 << endl;
			// for(int j = 0; j < total_points_cluster; j++)
			// {
			// 	out << "Point " << clusters[i].getPoint(j).getID() + 1 << ": ";
			// 	for(int p = 0; p < total_values; p++)
			// 		out << clusters[i].getPoint(j).getValue(p) << " ";

			// 	string point_name = clusters[i].getPoint(j).getName();

			// 	if(point_name != "")
			// 		out << "- " << point_name;

			// 	out << endl;
			// }

			out << "Cluster values: ";

			for(int j = 0; j < total_values; j++)
				out << clusters[i].getCentralValue(j) << " ";

			out << "\n\n";
		}
	}
};

int main(int argc, char *argv[])
{
	srand (42); // seed fixa para resultados reproduzíveis

	int total_points, total_values, K, max_iterations, has_name;

	cin >> total_points >> total_values >> K >> max_iterations >> has_name;

	vector<Point> points;
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

	KMeans kmeans(K, total_points, total_values, max_iterations);
	
	// Determina o arquivo de saída
	ofstream outfile;
	ostream* output = &cout;  // padrão é cout
	
	if(argc > 1) {
		outfile.open(argv[1]);
		if(outfile.is_open()) {
			output = &outfile;
		}
	}
	
	kmeans.run(points, *output);
	
	if(outfile.is_open()) {
		outfile.close();
	}

	return 0;
}