// kmeans_sequencial_mpi.cpp
// Versão adaptada do kmeans_sequencial.cpp para MPI + OpenMP (master-scatter approach).
// Mantém classes Point, Cluster e a lógica de saída do código sequencial.
// Compilar com: mpicxx -std=c++17 -fopenmp -O2 -o kmeans_mpi_openmp kmeans_sequencial_mpi.cpp -lm
//
// Execução:
// mpirun -np <P> ./kmeans_mpi_openmp [output_file] < input.txt
// (se mpirun -np 1 -> comportamento equivalente a "somente OpenMP" local)

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <string>
#include <mpi.h>
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
    Point(int id_point = -1, vector<double> values = vector<double>(), string name = "")
    {
        this->id_point = id_point;
        this->total_values = values.size();
        this->values = values;
        this->name = name;
        this->id_cluster = -1;
    }

    int getID() const { return id_point; }
    void setCluster(int id_cluster) { this->id_cluster = id_cluster; }
    int getCluster() const { return id_cluster; }
    double getValue(int index) const { return values[index]; }
    int getTotalValues() const { return total_values; }
    void addValue(double value) { values.push_back(value); }
    const vector<double>& getValues() const { return values; }
    string getName() const { return name; }
    void setValues(const vector<double>& v) { values = v; total_values = values.size(); }
};

class Cluster
{
private:
    int id_cluster;
    vector<double> central_values;
    vector<Point> points; // NOTE: in MPI mode, we avoid using points on non-master to reduce communication

public:
    Cluster(int id_cluster, Point point)
    {
        this->id_cluster = id_cluster;
        int total_values = point.getTotalValues();
        for (int i = 0; i < total_values; i++)
            central_values.push_back(point.getValue(i));
        points.push_back(point);
    }

    Cluster(int id_cluster, const vector<double> &centroid_values)
    {
        this->id_cluster = id_cluster;
        central_values = centroid_values;
    }

    void addPoint(const Point &point) { points.push_back(point); }

    bool removePoint(int id_point)
    {
        int total_points = points.size();
        for (int i = 0; i < total_points; i++)
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
    Point getPoint(int index) const { return points[index]; }
    int getTotalPoints() const { return points.size(); }
    int getID() const { return id_cluster; }

    const vector<double>& getCentralValues() const { return central_values; }
    void setCentralValues(const vector<double> &v) { central_values = v; }
};

class KMeans
{
private:
    int K; // number of clusters
    int total_values, total_points, max_iterations;
    vector<Cluster> clusters_seq; // used only for sequential printing if desired

    // helper: euclidean squared distance between centroid (flat pointer) and point values pointer
    static inline double dist_sq_centroid_point(const double* centroid, const double* pointvals, int D) {
        double s = 0.0;
        for (int j = 0; j < D; ++j) {
            double d = centroid[j] - pointvals[j];
            s += d * d;
        }
        return s;
    }

public:
    KMeans(int K, int total_points, int total_values, int max_iterations)
    {
        this->K = K;
        this->total_points = total_points;
        this->total_values = total_values;
        this->max_iterations = max_iterations;
    }

    // ---------- original sequential run (kept for reference / local runs) ----------
    void run_sequential(vector<Point> &points, ostream& out)
    {
        if (K > total_points) return;

        vector<int> prohibited_indexes;
        vector<Cluster> clusters;
        srand(42);

        // choose K distinct initial centers
        for (int i = 0; i < K; i++)
        {
            while (true)
            {
                int index_point = rand() % total_points;
                if (find(prohibited_indexes.begin(), prohibited_indexes.end(),
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
        while (true)
        {
            bool done = true;

            // associate each point to the nearest center
            for (int i = 0; i < total_points; i++)
            {
                int id_old_cluster = points[i].getCluster();

                // compute distance to cluster 0
                double sum = 0.0;
                for (int j = 0; j < total_values; j++)
                    sum += pow(clusters[0].getCentralValue(j) -
                               points[i].getValue(j), 2.0);
                double min_dist = sqrt(sum);
                int id_cluster_center = 0;

                for (int c = 1; c < K; c++)
                {
                    double sumc = 0.0;
                    for (int j = 0; j < total_values; j++)
                        sumc += pow(clusters[c].getCentralValue(j) -
                                    points[i].getValue(j), 2.0);
                    double dc = sqrt(sumc);
                    if (dc < min_dist)
                    {
                        min_dist = dc;
                        id_cluster_center = c;
                    }
                }

                if (id_old_cluster != id_cluster_center)
                {
                    if (id_old_cluster != -1)
                        clusters[id_old_cluster].removePoint(points[i].getID());

                    points[i].setCluster(id_cluster_center);
                    clusters[id_cluster_center].addPoint(points[i]);
                    done = false;
                }
            }

            // recalc centers
            for (int c = 0; c < K; c++)
            {
                int total_points_cluster = clusters[c].getTotalPoints();
                if (total_points_cluster > 0)
                {
                    for (int j = 0; j < total_values; j++)
                    {
                        double sum = 0.0;
                        for (int p = 0; p < total_points_cluster; p++)
                            sum += clusters[c].getPoint(p).getValue(j);
                        clusters[c].setCentralValue(j, sum / total_points_cluster);
                    }
                }
            }

            if (done || iter >= max_iterations)
            {
                out << "Break in iteration " << iter << "\n\n";
                break;
            }
            iter++;
        }

        // show elements
        for (int c = 0; c < K; c++)
        {
            int total_points_cluster = clusters[c].getTotalPoints();
            out << "Cluster " << clusters[c].getID() + 1 << endl;
            for (int j = 0; j < total_points_cluster; j++)
            {
                out << "Point " << clusters[c].getPoint(j).getID() + 1 << ": ";
                for (int p = 0; p < total_values; p++)
                    out << clusters[c].getPoint(j).getValue(p) << " ";
                string point_name = clusters[c].getPoint(j).getName();
                if (point_name != "") out << "- " << point_name;
                out << endl;
            }
            out << "Cluster values: ";
            for (int j = 0; j < total_values; j++)
                out << clusters[c].getCentralValue(j) << " ";
            out << "\n\n";
        }
    }

    // ---------- MPI + OpenMP master/worker run (master has all data, scatters partitions) ----------
    // points_master: on rank 0 contains full dataset; on other ranks is ignored (they'll receive flat_local).
    void run_mpi_master_scatter(vector<Point> &points_master, ostream& out)
    {
        int rank, P;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &P);

        // broadcast the basic header info is assumed already read by rank0 before calling this function,
        // but we have total_points and total_values stored in the object.
        // We'll compute distribution counts:
        int N = total_points;
        int D = total_values;

        // compute counts_points (how many points per rank) and displacements in points
        vector<int> counts_points(P, 0), displ_points(P, 0);
        int base = N / P;
        int rem = N % P;
        for (int r = 0; r < P; ++r) counts_points[r] = base + (r < rem ? 1 : 0);
        displ_points[0] = 0;
        for (int r = 1; r < P; ++r) displ_points[r] = displ_points[r-1] + counts_points[r-1];

        // prepare flattened data only on master
        vector<double> flat_all;
        if (rank == 0) {
            flat_all.resize((size_t)N * D);
            for (int i = 0; i < N; ++i) {
                const vector<double> &vals = points_master[i].getValues();
                for (int j = 0; j < D; ++j) flat_all[(size_t)i * D + j] = vals[j];
            }
        }

        // prepare sendcounts and displs (in doubles)
        vector<int> sendcounts_d(P, 0), displs_d(P, 0);
        for (int r = 0; r < P; ++r) {
            sendcounts_d[r] = counts_points[r] * D;
            displs_d[r] = displ_points[r] * D;
        }

        // each rank receives flat_local of size local_n * D
        int local_n = counts_points[rank];
        vector<double> flat_local((size_t)local_n * D);

        MPI_Scatterv(
            rank == 0 ? flat_all.data() : nullptr,
            sendcounts_d.data(),
            displs_d.data(),
            MPI_DOUBLE,
            flat_local.data(),
            local_n * D,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        // build local_points for this rank
        vector<Point> local_points;
        local_points.reserve(local_n);
        for (int i = 0; i < local_n; ++i) {
            vector<double> vals(D);
            for (int j = 0; j < D; ++j) vals[j] = flat_local[(size_t)i * D + j];
            int global_id = displ_points[rank] + i;
            local_points.emplace_back(global_id, vals);
        }

        // All ranks now have their local_points; master has points_master still.
        // Master chooses initial centroids and broadcasts them (flat K*D)
        vector<double> centroids((size_t)K * D, 0.0);
        if (rank == 0) {
            srand(42);
            vector<int> prohibited;
            for (int c = 0; c < K; ++c) {
                while (true) {
                    int idx = rand() % N;
                    if (find(prohibited.begin(), prohibited.end(), idx) == prohibited.end()) {
                        prohibited.push_back(idx);
                        // set centroid c as point idx
                        for (int j = 0; j < D; ++j)
                            centroids[c * D + j] = points_master[idx].getValue(j);
                        break;
                    }
                }
            }
        }
        MPI_Bcast(centroids.data(), K * D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // buffers for local accumulation
        vector<double> local_sum((size_t)K * D, 0.0);
        vector<int> local_count(K, 0);
        int iter = 0;

        // buffers on master for gathering contributions
        vector<double> gather_sums;
        vector<int> gather_counts;
        vector<int> gather_changed;
        if (rank == 0) {
            gather_sums.resize((size_t)P * K * D);
            gather_counts.resize((size_t)P * K);
            gather_changed.resize(P);
        }

        bool converged = false;

        while (true) {
            ++iter;
            // zero local accumulators
            std::fill(local_sum.begin(), local_sum.end(), 0.0);
            std::fill(local_count.begin(), local_count.end(), 0);
            int changed_local = 0;

            // Use OpenMP to process local points: each thread keeps private buffers to reduce atomics
            int nthreads = omp_get_max_threads();
            vector< vector<double> > thr_sum(nthreads, vector<double>((size_t)K * D, 0.0));
            vector< vector<int> > thr_count(nthreads, vector<int>(K, 0));
            vector<int> thr_changed(nthreads, 0);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_num_threads();
                // simple loop scheduling across threads
                for (int i = tid; i < (int)local_points.size(); i += nth) {
                    const vector<double>& pv = local_points[i].getValues();
                    // nearest centroid search (squared distance)
                    int best = 0;
                    double bestd = dist_sq_centroid_point(&centroids[0], pv.data(), D);
                    for (int c = 1; c < K; ++c) {
                        double d = dist_sq_centroid_point(&centroids[c * D], pv.data(), D);
                        if (d < bestd) { bestd = d; best = c; }
                    }
                    if (local_points[i].getCluster() != best) {
                        thr_changed[tid] += 1;
                        local_points[i].setCluster(best);
                    }
                    // accumulate into thread-local buffers
                    int basec = best * D;
                    for (int j = 0; j < D; ++j) thr_sum[tid][basec + j] += pv[j];
                    thr_count[tid][best] += 1;
                }
            } // end parallel

            // reduce thread buffers into local_sum/local_count
            for (int t = 0; t < nthreads; ++t) {
                for (int idx = 0; idx < K * D; ++idx) local_sum[idx] += thr_sum[t][idx];
                for (int c = 0; c < K; ++c) local_count[c] += thr_count[t][c];
                changed_local += thr_changed[t];
            }

            // Gather local_sum and local_count and changed flags to master
            MPI_Gather(local_sum.data(), K * D, MPI_DOUBLE,
                       rank == 0 ? gather_sums.data() : nullptr,
                       K * D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            MPI_Gather(local_count.data(), K, MPI_INT,
                       rank == 0 ? gather_counts.data() : nullptr,
                       K, MPI_INT, 0, MPI_COMM_WORLD);

            MPI_Gather(&changed_local, 1, MPI_INT,
                       rank == 0 ? gather_changed.data() : nullptr,
                       1, MPI_INT, 0, MPI_COMM_WORLD);

            // master aggregates and recomputes centroids
            if (rank == 0) {
                vector<double> global_sum((size_t)K * D, 0.0);
                vector<int> global_count(K, 0);
                for (int r = 0; r < P; ++r) {
                    double* src_sum = &gather_sums[(size_t)r * K * D];
                    for (int idx = 0; idx < K * D; ++idx) global_sum[idx] += src_sum[idx];
                    int* src_cnt = &gather_counts[(size_t)r * K];
                    for (int c = 0; c < K; ++c) global_count[c] += src_cnt[c];
                }
                // recompute centroids
                for (int c = 0; c < K; ++c) {
                    if (global_count[c] > 0) {
                        for (int j = 0; j < D; ++j) {
                            centroids[c * D + j] = global_sum[c * D + j] / (double)global_count[c];
                        }
                    } // else keep previous centroid (no points)
                }
                // decide convergence
                int sum_changed = 0;
                for (int r = 0; r < P; ++r) sum_changed += gather_changed[r];
                if (sum_changed == 0 || iter >= max_iterations) converged = true;
                else converged = false;
            }

            // broadcast updated centroids and convergence flag
            MPI_Bcast(centroids.data(), K * D, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            int conv_int = converged ? 1 : 0;
            MPI_Bcast(&conv_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
            converged = (conv_int == 1);

            if (converged) {
                if (rank == 0) out << "Break in iteration " << iter << "\n\n";
                break;
            }
        } // end while

        // gather final cluster assignments to master to print same format as sequential
        vector<int> local_assign(local_n, -1);
        for (int i = 0; i < local_n; ++i) local_assign[i] = local_points[i].getCluster();

        vector<int> recvcounts_assign(P), displs_assign(P);
        for (int r = 0; r < P; ++r) recvcounts_assign[r] = counts_points[r];
        displs_assign[0] = 0;
        for (int r = 1; r < P; ++r) displs_assign[r] = displs_assign[r-1] + recvcounts_assign[r-1];

        vector<int> all_assign;
        if (rank == 0) all_assign.resize(N);

        MPI_Gatherv(local_assign.data(), local_n, MPI_INT,
                    rank == 0 ? all_assign.data() : nullptr,
                    recvcounts_assign.data(), displs_assign.data(), MPI_INT,
                    0, MPI_COMM_WORLD);

        // master prints clusters and centroids similar to sequential output
        if (rank == 0) {
            // build clusters mapping
            vector<vector<int>> cluster_members(K);
            for (int gid = 0; gid < N; ++gid) {
                int cl = all_assign[gid];
                if (cl >= 0 && cl < K) cluster_members[cl].push_back(gid);
            }
            for (int c = 0; c < K; ++c) {
                out << "Cluster " << c + 1 << endl;
                for (int pid : cluster_members[c]) {
                    out << "Point " << pid + 1 << ": ";
                    const vector<double> &vals = points_master[pid].getValues();
                    for (int j = 0; j < D; ++j) out << vals[j] << " ";
                    out << endl;
                }
                out << "Cluster values: ";
                for (int j = 0; j < D; ++j) out << centroids[c * D + j] << " ";
                out << "\n\n";
            }
        }
    } // end run_mpi_master_scatter
};

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    srand(42); // fixed seed

    int total_points, total_values, K, max_iterations, has_name;
    // Only master needs to read whole input; still call cin on all ranks but only master will have real data
    if (rank == 0) {
        if (!(cin >> total_points >> total_values >> K >> max_iterations >> has_name)) {
            cerr << "Failed to read input header\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    // broadcast header to all ranks so they know sizes
    MPI_Bcast(&total_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_values, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&has_name, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<Point> points_master; // only filled on master
    if (rank == 0) {
        points_master.reserve(total_points);
        for (int i = 0; i < total_points; ++i) {
            vector<double> values;
            values.reserve(total_values);
            for (int j = 0; j < total_values; ++j) {
                double v; cin >> v;
                values.push_back(v);
            }
            if (has_name) {
                string nm; cin >> nm;
                points_master.emplace_back(i, values, nm);
            } else {
                points_master.emplace_back(i, values);
            }
        }
    } else {
        // non-master ranks still need to consume any input from stdin? No: master read all
        // No action needed here.
    }

    // Prepare output
    ofstream outfile;
    ostream *output = &cout;
    if (argc > 1) {
        if (rank == 0) {
            outfile.open(argv[1]);
            if (!outfile.is_open()) {
                cerr << "Error opening output file " << argv[1] << endl;
            } else {
                output = &outfile;
            }
        }
    }

    // Create KMeans and run Master-Scatter MPI+OpenMP version
    KMeans kmeans(K, total_points, total_values, max_iterations);
    kmeans.run_mpi_master_scatter(points_master, *output);

    if (outfile.is_open()) outfile.close();

    MPI_Finalize();
    return 0;
}
