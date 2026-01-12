#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <limits>
#include <eigen3/Eigen/Dense>

// Include OpenMP if available
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C" {
    /**
     * @brief Calculates ground resolution (mm/px) without allocating vectors.
     * Memory: O(1), Complexity: O(N)
     */
    double calculate_ground_resolution_c(
        double* u, double* v, double* z, int n,
        double fx, double fy, double cx, double cy) 
    {
        if (n == 0) return 0.0;

        // Indices for min/max x and y in 3D space
        int min_x_idx = 0, max_x_idx = 0;
        int min_y_idx = 0, max_y_idx = 0;
        
        // Track min/max values directly to avoid storing the 'x' and 'y' arrays
        double min_x_val = std::numeric_limits<double>::max();
        double max_x_val = -std::numeric_limits<double>::max();
        double min_y_val = std::numeric_limits<double>::max();
        double max_y_val = -std::numeric_limits<double>::max();

        // Single pass to find extremes
        // Note: Cannot easily parallelize finding ArgMin/ArgMax without reduction logic, 
        // but simple loop is very fast without allocation.
        for (int i = 0; i < n; ++i) {
            double zi = z[i];
            double xi = zi * (u[i] - cx) / fx;
            double yi = zi * (v[i] - cy) / fy;

            if (xi < min_x_val) { min_x_val = xi; min_x_idx = i; }
            if (xi > max_x_val) { max_x_val = xi; max_x_idx = i; }
            if (yi < min_y_val) { min_y_val = yi; min_y_idx = i; }
            if (yi > max_y_val) { max_y_val = yi; max_y_idx = i; }
        }

        // Calculate pixel resolutions (Euclidean distance in u,v plane)
        double du_x = u[max_x_idx] - u[min_x_idx];
        double dv_x = v[max_x_idx] - v[min_x_idx];
        double u_res = std::sqrt(du_x * du_x + dv_x * dv_x);

        double du_y = u[max_y_idx] - u[min_y_idx];
        double dv_y = v[max_y_idx] - v[min_y_idx];
        double v_res = std::sqrt(du_y * du_y + dv_y * dv_y);

        // Calculate spatial resolutions (Euclidean distance in x,y plane)
        double dx_x = max_x_val - min_x_val;
        double dy_x = (z[max_x_idx] * (v[max_x_idx] - cy) / fy) - (z[min_x_idx] * (v[min_x_idx] - cy) / fy);
        double x_res = std::sqrt(dx_x * dx_x + dy_x * dy_x);

        double dx_y = (z[max_y_idx] * (u[max_y_idx] - cx) / fx) - (z[min_y_idx] * (u[min_y_idx] - cx) / fx);
        double dy_y = max_y_val - min_y_val;
        double y_res = std::sqrt(dx_y * dx_y + dy_y * dy_y);

        if (u_res <= 1e-9 || v_res <= 1e-9) return 0.0;

        return (x_res / u_res + y_res / v_res) / 2.0;
    }

    /**
     * @brief Calculates slant angle using Normal Equations (A^T * A * x = A^T * b).
     * Replaces the O(N) memory matrix with a fixed 9x9 accumulator.
     */
    void calculate_slant_c(
        double* depth_data, int rows, int cols,
        double fx, double fy, double cx, double cy,
        double* out_slant_angle) 
    {
        // 9x9 Matrix for A^T * A
        Eigen::Matrix<double, 9, 9> AtA = Eigen::Matrix<double, 9, 9>::Zero();
        // 9x1 Vector for A^T * b
        Eigen::Matrix<double, 9, 1> Atb = Eigen::Matrix<double, 9, 1>::Zero();

        int n_points = 0;

        // Parallel accumulation requires handling shared state. 
        // For simplicity and thread-safety without heavy mutexes, we use sequential here 
        // OR we would need thread-local matrices and reduce them.
        // Given the speed of 9x9 math, sequential accumulation is usually memory-bound anyway.
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                double z = depth_data[r * cols + c];
                if (z <= 0) continue;

                double x = z * (c - cx) / fx;
                double y = z * (r - cy) / fy;

                // Basis functions for 3rd order polynomial
                double p[9];
                p[0] = 1.0;
                p[1] = x;
                p[2] = y;
                p[3] = x * x;
                p[4] = x * y;
                p[5] = y * y;
                p[6] = x * x * y;
                p[7] = x * y * y;
                p[8] = y * y * y;

                // Update Normal Equations
                // AtA += p * p^T
                // Atb += p * z
                for (int i = 0; i < 9; ++i) {
                    Atb(i) += p[i] * z;
                    for (int j = i; j < 9; ++j) {
                        double val = p[i] * p[j];
                        AtA(i, j) += val;
                        if (i != j) AtA(j, i) += val; // Symmetric
                    }
                }
                n_points++;
            }
        }

        if (n_points < 9) {
            *out_slant_angle = 0.0;
            return;
        }

        // Solve 9x9 system (very fast)
        Eigen::VectorXd coeffs = AtA.ldlt().solve(Atb);

        double n_x = coeffs(1);
        double n_y = coeffs(2);
        double n_z = -1.0;

        double normal_magnitude = std::sqrt(n_x * n_x + n_y * n_y + n_z * n_z);
        *out_slant_angle = std::acos(n_z / (normal_magnitude + 1e-9)) * 180.0 / M_PI;
    }

    /**
     * @brief Calculates UCIQE.
     * Optimizations: O(N) Histogram for contrast instead of O(N log N) Sort.
     * Parallelized statistics calculation.
     */
    void calculate_uciqe_c(
        unsigned char* lab_data, int rows, int cols,
        double* out_uciqe)
    {
        int n = rows * cols;
        if (n == 0) { *out_uciqe = 0.0; return; }

        double chroma_sum = 0.0;
        double chroma_sq_sum = 0.0;
        double satur_sum = 0.0;
        long hist[256] = {0}; // Histogram for L channel

        #pragma omp parallel
        {
            double local_c_sum = 0.0;
            double local_c_sq_sum = 0.0;
            double local_s_sum = 0.0;
            long local_hist[256] = {0};

            #pragma omp for nowait
            for (int i = 0; i < n; ++i) {
                int idx = i * 3;
                double L = static_cast<double>(lab_data[idx]);
                double A = static_cast<double>(lab_data[idx + 1]);
                double B = static_cast<double>(lab_data[idx + 2]);

                // Binning for L (avoid branching logic if possible)
                local_hist[lab_data[idx]]++;

                // Chroma
                double c = std::sqrt(A*A + B*B);
                local_c_sum += c;
                local_c_sq_sum += c * c;

                // Saturation
                if (L > 0) {
                    local_s_sum += c / L;
                }
            }

            #pragma omp critical
            {
                chroma_sum += local_c_sum;
                chroma_sq_sum += local_c_sq_sum;
                satur_sum += local_s_sum;
                for(int k=0; k<256; k++) hist[k] += local_hist[k];
            }
        }

        // 1. Chroma Std Dev
        double chroma_mean = chroma_sum / n;
        double variance = (chroma_sq_sum / n) - (chroma_mean * chroma_mean);
        double sc = std::sqrt(std::max(0.0, variance));

        // 2. Luminance Contrast (Histogram Method)
        int top_k = static_cast<int>(0.01 * n);
        if (top_k == 0) top_k = 1;

        // Find sum of bottom 1%
        double bottom_sum = 0.0;
        int count = 0;
        for (int i = 0; i < 256; ++i) {
            if (count + hist[i] >= top_k) {
                int needed = top_k - count;
                bottom_sum += needed * i;
                break;
            }
            bottom_sum += hist[i] * i;
            count += hist[i];
        }

        // Find sum of top 1%
        double top_sum = 0.0;
        count = 0;
        for (int i = 255; i >= 0; --i) {
            if (count + hist[i] >= top_k) {
                int needed = top_k - count;
                top_sum += needed * i;
                break;
            }
            top_sum += hist[i] * i;
            count += hist[i];
        }

        double conl = (top_sum / top_k) - (bottom_sum / top_k);

        // 3. Saturation
        double satur = satur_sum / n;

        *out_uciqe = 0.4680 * sc + 0.2745 * conl + 0.2576 * satur;
    }

    /**
     * @brief Calculates EME/LogAMEE.
     * Parallelized block loops.
     */
    void calculate_channel_eme_c(
        unsigned char* ch_data, int rows, int cols, 
        int blocksize, bool is_logamee, 
        double gamma, double k,
        double* out_eme)
    {
        int num_x = (rows + blocksize - 1) / blocksize;
        int num_y = (cols + blocksize - 1) / blocksize;

        if (num_x == 0 || num_y == 0) { *out_eme = 0.0; return; }

        double eme_sum = 0.0;
        double w = 1.0 / (num_x * num_y);

        // Parallelize outer loop
        #pragma omp parallel for reduction(+:eme_sum)
        for (int i = 0; i < num_x; ++i) {
            int xlb = i * blocksize;
            int xrb = std::min((i + 1) * blocksize, rows);

            for (int j = 0; j < num_y; ++j) {
                int ylb = j * blocksize;
                int yrb = std::min((j + 1) * blocksize, cols);

                unsigned char blockmin = 255;
                unsigned char blockmax = 0;

                // Scan block
                for (int r = xlb; r < xrb; ++r) {
                    const unsigned char* row_ptr = &ch_data[r * cols];
                    for (int c = ylb; c < yrb; ++c) {
                        unsigned char val = row_ptr[c];
                        if (val < blockmin) blockmin = val;
                        if (val > blockmax) blockmax = val;
                    }
                }

                double fmin = static_cast<double>(blockmin);
                double fmax = static_cast<double>(blockmax);

                if (is_logamee) {
                    if (k - fmin == 0) continue;
                    double top = k * (fmax - fmin) / (k - fmin);
                    double bottom = fmax + fmin - fmax * fmin / gamma;
                    if (std::abs(bottom) < 1e-6) continue;
                    
                    double m = top / bottom;
                    if (m > 1e-9) {
                        eme_sum += m * std::log(m);
                    }
                } else {
                    if (fmin < 1e-9) fmin = 1.0;
                    if (fmax < 1e-9) fmax = 1.0; 
                    eme_sum += 2 * w * std::log(fmax / fmin);
                }
            }
        }

        if (is_logamee) {
            *out_eme = gamma - gamma * std::pow(1.0 - eme_sum / gamma, w);
        } else {
            *out_eme = eme_sum;
        }
    }

    /**
     * @brief Calculates UICM. 
     * Kept simple as the heavy lifting (sorting) happens outside or is fast enough.
     */
    void calculate_uicm_c(
        double* rgl_trimmed, double* ybl_trimmed, int n, 
        double* out_uicm)
    {
        if (n == 0) { *out_uicm = 0.0; return; }

        // Use array mapping for Eigen-like vectorization if desired, 
        // but simple loops are often auto-vectorized by -O3.
        double sum_rg = 0, sum_yb = 0;
        double sq_sum_rg = 0, sq_sum_yb = 0;

        #pragma omp parallel for reduction(+:sum_rg, sum_yb, sq_sum_rg, sq_sum_yb)
        for(int i=0; i<n; ++i) {
            double r = rgl_trimmed[i];
            double y = ybl_trimmed[i];
            sum_rg += r;
            sum_yb += y;
            sq_sum_rg += r*r;
            sq_sum_yb += y*y;
        }

        double urg = sum_rg / n;
        double uyb = sum_yb / n;
        
        double var_rg = (sq_sum_rg / n) - (urg * urg);
        double var_yb = (sq_sum_yb / n) - (uyb * uyb);
        
        double s2rg = std::max(0.0, var_rg);
        double s2yb = std::max(0.0, var_yb);

        double norm = std::sqrt(urg * urg + uyb * uyb);
        double s_sum_sqrt = std::sqrt(s2rg + s2yb);
        *out_uicm = -0.0268 * norm + 0.1586 * s_sum_sqrt;
    }

} // extern "C"