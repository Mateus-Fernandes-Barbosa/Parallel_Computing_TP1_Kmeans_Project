// kmeans_mpi_openmp_fixed.cpp
// compile:
// g++ -std=c++17 -fopenmp -o kmeans_mpi_openmp_fixed kmeans_mpi_openmp_fixed.cpp -lm
// run (example):
// mpirun -np 4 ./kmeans_openmp < input.txt > output_openmp.txt

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <omp.h>
#include <mpi.h>

using namespace std;

class Point
{
private:
    int id_point;
    int id_cluster;
    vector<double> values;
    string name;

public:
    Point() : id_point(-1), id_cluster(-1), values(), name("") {}
    Point(int id_point, const vector<double>& values, const string& name = "")
    {
        this->id_point = id_point;
        this->values = values;
        this->name = name;
        id_cluster = -1;
    }

    int getID() const { return id_point; }
    void setCluster(int id_cluster) { this->id_cluster = id_cluster; }
    int getCluster() const { return id_cluster; }
    double getValue(int index) const { return values[index]; }
    int getTotalValues() const { return (int)values.size(); }
    const vector<double>& getValues() const { return values; }
    string getName() const { return name; }
};

class Cluster
{
private:
    int id_cluster;
    vector<double> central_values;
    vector<Point> points; // only root will contain the full set of assigned Points

public:
    Cluster() : id_cluster(-1), central_values(), points() {}

    // initialize cluster centroid from a vector<double>
    Cluster(int id_cluster, const vector<double>& center_values)
    {
        this->id_cluster = id_cluster;
        this->central_values = center_values;
    }

    void addPoint(const Point& point)
    {
        points.push_back(point);
    }

    bool removePoint(int id_point)
    {
        for (size_t i = 0; i < points.size(); ++i)
        {
            if (points[i].getID() == id_point)
            {
                points.erase(points.begin() + i);
                return true;
            }
        }
        return false;
    }

    double getCentralValue(int index) const { return central_values[index]; }
    void setCentralValue(int index, double value) { central_values[index] = value; }
    const vector<double>& getCentralValues() const { return central_values; }

    Point getPoint(int index) const { return points[index]; }
    int getTotalPoints() const { return (int)points.size(); }
    int getID() const { return id_cluster; }

    // recompute centroid from owned points (root uses this after gathering membership)
    void recomputeCentroid(int total_values)
    {
        vector<double> sums(total_values, 0.0);
        if (points.empty()) return; // keep current centroid if empty (alternatively randomize)
        for (const auto& p : points)
        {
            for (int j = 0; j < total_values; ++j)
                sums[j] += p.getValue(j);
        }
        for (int j = 0; j < total_values; ++j)
            central_values[j] = sums[j] / (double)points.size();
    }

    void clearPoints() { points.clear(); }
};

class KMeans
{
private:
    const int K;
    const int total_points;
    const int total_values;
    const int max_iterations;
    vector<Cluster> clusters; // only root keeps full clusters' assigned points

    // compute nearest centroid id using OpenMP + SIMD-friendly loop
    int parallel_getIDNearestCenter(const Point& point)
    {
        vector<double> distances(K);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < K; ++i)
        {
            double sum = 0.0;
            const vector<double>& center = clusters[i].getCentralValues();
            // avoid pow for squared difference
            #pragma omp simd reduction(+:sum)
            for (int j = 0; j < total_values; ++j)
            {
                double diff = center[j] - point.getValue(j);
                sum += diff * diff;
            }
            distances[i] = sqrt(sum);
        }

        double min_dist = distances[0];
        int id_cluster_center = 0;
        for (int i = 1; i < K; ++i)
        {
            if (distances[i] < min_dist)
            {
                min_dist = distances[i];
                id_cluster_center = i;
            }
        }
        return id_cluster_center;
    }

    // Gather cluster membership lists (global) to root.
    // local_hist: vector of K vectors containing global point indices that must be added/removed.
    // Returns global_hist on root (vector of K vectors), on non-root returns empty vector.
    vector<vector<int>> converge_kluster_history(const vector<vector<int>>& local_hist, int rank, int P)
    {
        // Prepare send counts (number of ints per cluster) - gather them first
        vector<int> send_counts(K);
        for (int k = 0; k < K; ++k) send_counts[k] = (int)local_hist[k].size();

        vector<int> all_sizes; // only on root
        if (rank == 0) all_sizes.resize(P * K);

        // Each process sends K ints (counts per cluster). Root receives P*K ints.
        MPI_Gather(send_counts.data(), K, MPI_INT,
                   rank == 0 ? all_sizes.data() : nullptr, K, MPI_INT,
                   0, MPI_COMM_WORLD);

        if (rank != 0)
        {
            // Non-root: participate in the Gathervs, return empty
            for (int k = 0; k < K; ++k)
            {
                // send local_hist[k] to root, root will provide recvcounts/displs
                MPI_Gatherv(local_hist[k].empty() ? nullptr : const_cast<int*>(local_hist[k].data()),
                            (int)local_hist[k].size(), MPI_INT,
                            nullptr, nullptr, nullptr, MPI_INT,
                            0, MPI_COMM_WORLD);
            }
            return {};
        }
        else
        {
            // Root: prepare recvcounts/displacements per cluster and gather
            vector<vector<int>> global_hist(K);

            // For each cluster k, determine recvcounts per process and total size
            for (int k = 0; k < K; ++k)
            {
                vector<int> recv_counts(P);
                vector<int> displs(P);
                int total_elems = 0;
                for (int p = 0; p < P; ++p)
                {
                    recv_counts[p] = all_sizes[p * K + k];
                    total_elems += recv_counts[p];
                }
                global_hist[k].resize(total_elems);
                // build displs
                displs[0] = 0;
                for (int p = 1; p < P; ++p) displs[p] = displs[p - 1] + recv_counts[p - 1];

                // Now gather the actual integer arrays for this cluster
                MPI_Gatherv(local_hist[k].empty() ? nullptr : const_cast<int*>(local_hist[k].data()),
                            (int)local_hist[k].size(), MPI_INT,
                            global_hist[k].data(),
                            recv_counts.data(),
                            displs.data(),
                            MPI_INT,
                            0, MPI_COMM_WORLD);
            }
            return global_hist;
        }
    }

public:
    KMeans(int K, int total_points, int total_values, int max_iterations)
        : K(K), total_points(total_points), total_values(total_values), max_iterations(max_iterations) {}

    // points_full: only root must pass the full points vector. Non-root should pass an empty vector.
    void run_mpi(vector<Point>& points_full, ostream& out)
    {
        if (K > total_points) return;

        int rank, P;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &P);

        // Determine distribution of points among processes (in point units)
        int points_per_process = total_points / P;
        int remainder = total_points % P;
        vector<int> counts_points(P);
        vector<int> displs_points(P);
        int cur = 0;
        for (int i = 0; i < P; ++i)
        {
            int cnt = points_per_process + (i < remainder ? 1 : 0);
            counts_points[i] = cnt;
            displs_points[i] = cur;
            cur += cnt;
        }

        // Pack flat point attributes on root and broadcast to all processes
        vector<double> flat_points(total_points * total_values);
        if (rank == 0)
        {
            for (int i = 0; i < total_points; ++i)
            {
                const vector<double>& vals = points_full[i].getValues();
                for (int j = 0; j < total_values; ++j)
                    flat_points[i * total_values + j] = vals[j];
            }
        }
        // Broadcast the flattened points array to all ranks
        MPI_Bcast(flat_points.data(), total_points * total_values, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Reconstruct local_points from flat_points relying on counts/displs
        int local_n_points = counts_points[rank];
        int start_global_point_idx = displs_points[rank];
        vector<Point> local_points;
        local_points.reserve(local_n_points);
        for (int i = 0; i < local_n_points; ++i)
        {
            int global_idx = start_global_point_idx + i;
            vector<double> vals(total_values);
            for (int j = 0; j < total_values; ++j)
                vals[j] = flat_points[global_idx * total_values + j];
            // name is only available on root; non-root leave empty
            string name = (rank == 0 ? points_full[global_idx].getName() : "");
            local_points.emplace_back(global_idx, vals, name);
        }

        // INITIAL CENTROIDS: root chooses K random indices and broadcasts centroid values.
        vector<double> flat_centroids(K * total_values);
        if (rank == 0)
        {
            // select K distinct random points as initial centroids
            srand(42); // deterministic seed on root
            vector<int> chosen;
            chosen.reserve(K);
            while ((int)chosen.size() < K)
            {
                int idx = rand() % total_points;
                if (find(chosen.begin(), chosen.end(), idx) == chosen.end())
                    chosen.push_back(idx);
            }
            // initialize clusters on root and fill flat_centroids
            clusters.clear();
            for (int i = 0; i < K; ++i)
            {
                vector<double> center_vals = points_full[chosen[i]].getValues();
                clusters.emplace_back(i, center_vals);
                // ensure the root cluster's points vector contains the chosen point as initial member
                clusters[i].clearPoints();
                clusters[i].addPoint(points_full[chosen[i]]);
                for (int j = 0; j < total_values; ++j)
                    flat_centroids[i * total_values + j] = center_vals[j];
            }
        }

        // Broadcast initial centroids' flat values
        MPI_Bcast(flat_centroids.data(), K * total_values, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // On non-root, create clusters with those centroids
        if (rank != 0)
        {
            clusters.clear();
            for (int i = 0; i < K; ++i)
            {
                vector<double> center_vals(total_values);
                for (int j = 0; j < total_values; ++j)
                    center_vals[j] = flat_centroids[i * total_values + j];
                clusters.emplace_back(i, center_vals);
            }
        }

        // MAIN ITERATION LOOP
        int iter = 1;
        while (true)
        {
            bool local_changed = false;

            // local lists for additions/removals (global indices)
            vector<vector<int>> local_cluster_add(K);
            vector<vector<int>> local_cluster_remove(K);

            // For each local point find nearest centroid
            for (int i = 0; i < (int)local_points.size(); ++i)
            {
                Point& cur_point = local_points[i];
                int old_cluster = cur_point.getCluster();
                int best = parallel_getIDNearestCenter(cur_point);
                if (best != old_cluster)
                {
                    int global_index = cur_point.getID();
                    local_cluster_add[best].push_back(global_index);
                    if (old_cluster != -1)
                        local_cluster_remove[old_cluster].push_back(global_index);
                    cur_point.setCluster(best); // update local copy for future iterations
                    local_changed = true;
                }
            }

            // Allreduce to see if ANY process changed something
            int local_done = local_changed ? 0 : 1; // 1 means this rank did not change
            int global_done;
            MPI_Allreduce(&local_done, &global_done, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            bool done = (global_done == 1);

            // Gather "add" and "remove" lists to root (each returns a vector<vector<int>> on root)
            vector<vector<int>> global_cluster_add = converge_kluster_history(local_cluster_add, rank, P);
            vector<vector<int>> global_cluster_remove = converge_kluster_history(local_cluster_remove, rank, P);

            // Root updates clusters membership
            if (rank == 0)
            {
                // First remove members indicated in global_cluster_remove
                for (int k = 0; k < K; ++k)
                {
                    for (int idx : global_cluster_remove[k])
                        clusters[k].removePoint(idx);
                }
                // Then add members indicated in global_cluster_add
                for (int k = 0; k < K; ++k)
                {
                    for (int gidx : global_cluster_add[k])
                    {
                        clusters[k].addPoint(points_full[gidx]);
                    }
                }

                // Recompute centroids at root based on assigned points
                for (int k = 0; k < K; ++k)
                {
                    if (clusters[k].getTotalPoints() > 0)
                        clusters[k].recomputeCentroid(total_values);
                    // If cluster is empty, we keep it unchanged (alternatively reinitialize)
                    const vector<double>& center = clusters[k].getCentralValues();
                    for (int j = 0; j < total_values; ++j)
                        flat_centroids[k * total_values + j] = center[j];
                }
            }

            // Broadcast new centroids to all ranks
            MPI_Bcast(flat_centroids.data(), K * total_values, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Workers update centroid values
            if (rank != 0)
            {
                for (int k = 0; k < K; ++k)
                {
                    for (int j = 0; j < total_values; ++j)
                    {
                        clusters[k].setCentralValue(j, flat_centroids[k * total_values + j]);
                    }
                }
            }
            else
            {
                // Root also update local cluster central values (already in flat_centroids)
                for (int k = 0; k < K; ++k)
                    for (int j = 0; j < total_values; ++j)
                        clusters[k].setCentralValue(j, flat_centroids[k * total_values + j]);
            }

            if (done || iter >= max_iterations)
            {
                out << "Break in iteration " << iter << "\n\n";
                break;
            }
            iter++;
        }

        // Only root prints final cluster assignments and centroids
        if (rank == 0)
        {
            for (int i = 0; i < K; ++i)
            {
                out << "Cluster " << clusters[i].getID() + 1 << endl;
                int total_points_cluster = clusters[i].getTotalPoints();
                for (int j = 0; j < total_points_cluster; ++j)
                {
                    Point p = clusters[i].getPoint(j);
                    out << "Point " << p.getID() + 1 << ": ";
                    for (int v = 0; v < total_values; ++v)
                        out << p.getValue(v) << " ";
                    if (p.getName() != "")
                        out << "- " << p.getName();
                    out << "\n";
                }
                out << "Cluster values: ";
                for (int v = 0; v < total_values; ++v)
                    out << clusters[i].getCentralValue(v) << " ";
                out << "\n\n";
            }
        }
    }
};

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int total_points, total_values, K, max_iterations, has_name;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vector<Point> points; // only root will keep full points

    if (rank == 0)
    {
        // root reads input
        if (!(cin >> total_points >> total_values >> K >> max_iterations >> has_name))
        {
            cerr << "Failed to read input header\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        points.reserve(total_points);
        for (int i = 0; i < total_points; ++i)
        {
            vector<double> values(total_values);
            for (int j = 0; j < total_values; ++j)
                cin >> values[j];
            if (has_name)
            {
                string name;
                cin >> ws;
                cin >> name;
                points.emplace_back(i, values, name);
            }
            else
            {
                points.emplace_back(i, values);
            }
        }
    }

    // Broadcast header info to all ranks (all ranks need total_points/values/etc)
    MPI_Bcast(&total_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_values, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&has_name, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // KMeans object created on all ranks
    KMeans kmeans(K, total_points, total_values, max_iterations);

    // Prepare output stream (root can optionally write to file via argv argument)
    ofstream outfile;
    ostream* output = &cout;
    if (rank == 0 && argc > 1)
    {
        outfile.open(argv[1]);
        if (outfile.is_open()) output = &outfile;
    }

    // Run algorithm; points vector passed only on root
    kmeans.run_mpi(points, *output);

    if (rank == 0 && outfile.is_open()) outfile.close();

    MPI_Finalize();
    return 0;
}
