#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <eigen3/Eigen/Dense>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Helper: One-Pass Statistics (Mean, Std, Skew, Kurt) ---
struct Stats {
    double mean;
    double std_dev;
    double skew;
    double kurtosis;
};

// Calculates stats using raw moments to avoid storing data
// S1 = sum(x), S2 = sum(x^2), S3 = sum(x^3), S4 = sum(x^4)
inline Stats calculate_stats_moments(double s1, double s2, double s3, double s4, size_t n_size) {
    if (n_size < 2) return {0,0,0,0};
    double n = (double)n_size;

    double mean = s1 / n;
    
    // Variance (population)
    double var = (s2 / n) - (mean * mean);
    if (var < 1e-9) return {mean, 0.0, 0.0, 0.0};
    
    double std_dev = std::sqrt(var);

    // Skewness
    // M3 = E[x^3] - 3*mu*sigma^2 - mu^3
    // Simplified using raw moments:
    double m3 = (s3 / n) - 3.0 * mean * (s2 / n) + 2.0 * mean * mean * mean;
    double skew = m3 / (var * std_dev);

    // Kurtosis (Fisher)
    // M4 = E[x^4] - 4*mu*E[x^3] + 6*mu^2*E[x^2] - 3*mu^4
    double m4 = (s4 / n) - 4.0 * mean * (s3 / n) + 6.0 * mean * mean * (s2 / n) - 3.0 * mean * mean * mean * mean;
    double kurtosis = (m4 / (var * var)) - 3.0;

    return {mean, std_dev, skew, kurtosis};
}

inline int clamp_idx(int idx, int max_size) {
    if (idx < 0) return 0;
    if (idx >= max_size) return max_size - 1;
    return idx;
}

// Templated HOG with One-Pass Stats
template <typename T>
void compute_hog_impl(
    const T* img_data, int rows, int cols, int n_bins,
    double* out_features)
{
    // Clear output
    int stats_offset = n_bins;
    std::memset(out_features, 0, (n_bins + 4) * sizeof(double));

    double bin_width = 180.0 / n_bins;
    double total_mag_sum = 0.0;
    
    // Raw moments accumulators
    double s1 = 0, s2 = 0, s3 = 0, s4 = 0;
    int n_pixels = rows * cols;

    // Use OpenMP to accumulate histogram? 
    // HOG is often run on small patches (e.g. 16x16 or 64x64). Threading overhead might hurt.
    // We stick to serial for patch-level ops, assuming caller threads patches.
    
    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            int idx_c = r * cols + c;
            
            // Central Difference
            double val_r = (double)img_data[r * cols + clamp_idx(c + 1, cols)];
            double val_l = (double)img_data[r * cols + clamp_idx(c - 1, cols)];
            double dx = val_r - val_l;

            double val_d = (double)img_data[clamp_idx(r + 1, rows) * cols + c];
            double val_u = (double)img_data[clamp_idx(r - 1, rows) * cols + c];
            double dy = val_d - val_u;

            double mag = std::sqrt(dx * dx + dy * dy);
            double ang = std::atan2(dy, dx) * 180.0 / M_PI;

            if (ang < 0) ang += 180.0;
            if (ang >= 180.0) ang -= 180.0;

            int bin = (int)(ang / bin_width);
            if (bin >= n_bins) bin = n_bins - 1;

            out_features[bin] += mag;
            total_mag_sum += mag;

            // Stats accumulation
            double m2 = mag * mag;
            s1 += mag;
            s2 += m2;
            s3 += m2 * mag;
            s4 += m2 * m2;
        }
    }

    // Normalize Histogram
    if (total_mag_sum > 1e-9) {
        double inv_sum = 1.0 / total_mag_sum;
        for(int i = 0; i < n_bins; ++i) out_features[i] *= inv_sum;
    }

    Stats st = calculate_stats_moments(s1, s2, s3, s4, n_pixels);
    out_features[stats_offset + 0] = st.mean;
    out_features[stats_offset + 1] = st.std_dev;
    out_features[stats_offset + 2] = st.skew;
    out_features[stats_offset + 3] = st.kurtosis;
}

extern "C" {

    void extract_color_features_c(
        unsigned char* img_data, int rows, int cols, 
        int bins, int range_min, int range_max,
        double* out_features)
    {
        std::memset(out_features, 0, bins * sizeof(double));
        int num_pixels = rows * cols;
        float range_width = (float)(range_max - range_min);
        float scale = (range_width > 0) ? bins / range_width : 1.0f;

        for (int i = 0; i < num_pixels; ++i) {
            int val = img_data[i];
            if (val >= range_min && val < range_max) {
                int bin_idx = (int)((val - range_min) * scale);
                if (bin_idx >= bins) bin_idx = bins - 1; 
                out_features[bin_idx]++;
            }
        }

        double sum = 0.0;
        for (int i = 0; i < bins; ++i) sum += out_features[i];
        if (sum > 0) {
            double inv_sum = 1.0 / sum;
            for (int i = 0; i < bins; ++i) out_features[i] *= inv_sum;
        }
    }

    void extract_lbp_features_c(
        unsigned char* gray_data, int rows, int cols, int n_points,
        double* out_features)
    {
        int n_bins = n_points + 2;
        std::memset(out_features, 0, n_bins * sizeof(double));

        // Optimization: Use pointers to avoid repeated multiplication
        for (int i = 1; i < rows - 1; ++i) {
            const unsigned char* p_prev = &gray_data[(i-1) * cols];
            const unsigned char* p_curr = &gray_data[i * cols];
            const unsigned char* p_next = &gray_data[(i+1) * cols];
            
            for (int j = 1; j < cols - 1; ++j) {
                unsigned char c = p_curr[j];
                int code = 0;
                
                // Unrolled bit shifting
                if (p_prev[j-1] >= c) code |= 128;
                if (p_prev[j]   >= c) code |= 64;
                if (p_prev[j+1] >= c) code |= 32;
                if (p_curr[j+1] >= c) code |= 16;
                if (p_next[j+1] >= c) code |= 8;
                if (p_next[j]   >= c) code |= 4;
                if (p_next[j-1] >= c) code |= 2;
                if (p_curr[j-1] >= c) code |= 1;

                // Simple uniform mapping approximation or raw binning
                // Original code mapped 0-255 to n_bins
                int bin_idx = (code * n_bins) >> 8; // approx /256
                if (bin_idx >= n_bins) bin_idx = n_bins - 1;
                out_features[bin_idx]++;
            }
        }

        double sum = 0.0;
        for (int i = 0; i < n_bins; ++i) sum += out_features[i];
        if (sum > 0) {
            double inv = 1.0 / sum;
            for (int i = 0; i < n_bins; ++i) out_features[i] *= inv;
        }
    }

    void extract_glcm_features_c(
        unsigned char* img_data, int rows, int cols,
        double* out_features)
    {
        // Allocate histogram once. 256*256 doubles = 512KB. 
        // Allocating on heap is safer than stack.
        std::vector<double> glcm(256 * 256); 
        
        int drs[] = {0, -1, -1, -1};
        int dcs[] = {1, 1, 0, -1};
        int feat_idx = 0;

        for (int k = 0; k < 4; ++k) {
            int dr = drs[k];
            int dc = dcs[k];

            // Fast clear
            std::fill(glcm.begin(), glcm.end(), 0.0);
            double total_sum = 0.0;

            int r_start = (dr < 0) ? -dr : 0;
            int r_end   = (dr > 0) ? rows - dr : rows;
            int c_start = (dc < 0) ? -dc : 0;
            int c_end   = (dc > 0) ? cols - dc : cols;

            // Compute GLCM
            for (int r = r_start; r < r_end; ++r) {
                const unsigned char* row_ptr = &img_data[r * cols];
                const unsigned char* next_row_ptr = &img_data[(r + dr) * cols];
                
                for (int c = c_start; c < c_end; ++c) {
                    unsigned char src = row_ptr[c];
                    unsigned char dst = next_row_ptr[c + dc];
                    glcm[src * 256 + dst]++;
                    total_sum++;
                }
            }

            if (total_sum == 0) {
                for(int z=0; z<6; ++z) out_features[feat_idx++] = 0.0;
                continue;
            }

            double norm = 1.0 / (2.0 * total_sum);
            double contrast = 0, dissimilarity = 0, homogeneity = 0, asm_val = 0;
            double mean_i = 0, mean_j = 0;

            // Iterate only non-zero entries? Dense iteration is 65k ops. 
            // For typical photos, sparse map is faster, but dense is constant time. 
            // Stick to dense for SIMD friendliness if compiler optimizes.
            for(int i=0; i<256; ++i) {
                const double* row_glcm = &glcm[i * 256];
                for(int j=0; j<256; ++j) {
                    double p = (row_glcm[j] + glcm[j * 256 + i]) * norm;
                    if (p > 1e-12) {
                        double diff = (double)(i - j);
                        double diff_sq = diff * diff;
                        contrast += p * diff_sq;
                        dissimilarity += p * std::abs(diff);
                        homogeneity += p / (1.0 + diff_sq);
                        asm_val += p * p;
                        mean_i += i * p;
                        mean_j += j * p;
                    }
                }
            }

            double energy = std::sqrt(asm_val);
            double var_i = 0, var_j = 0, cov = 0;
            
            for(int i=0; i<256; ++i) {
                const double* row_glcm = &glcm[i * 256];
                for(int j=0; j<256; ++j) {
                    double p = (row_glcm[j] + glcm[j * 256 + i]) * norm;
                    if (p > 1e-12) {
                        double di = i - mean_i;
                        double dj = j - mean_j;
                        var_i += p * di * di;
                        var_j += p * dj * dj;
                        cov   += p * di * dj;
                    }
                }
            }
            
            double std_prod = std::sqrt(var_i * var_j);
            double correlation = (std_prod > 1e-10) ? cov / std_prod : 0.0;

            out_features[feat_idx++] = contrast;
            out_features[feat_idx++] = dissimilarity;
            out_features[feat_idx++] = homogeneity;
            out_features[feat_idx++] = energy;
            out_features[feat_idx++] = asm_val;
            out_features[feat_idx++] = correlation;
        }
    }

    void extract_hog_features_c(
        unsigned char* img_data, int rows, int cols, int n_bins,
        double* out_features)
    {
        compute_hog_impl<unsigned char>(img_data, rows, cols, n_bins, out_features);
    }

    void extract_hog_features_depth_c(
        double* depth_data, int rows, int cols, int n_bins,
        double* out_features)
    {
        compute_hog_impl<double>(depth_data, rows, cols, n_bins, out_features);
    }

    // Optimized Principal Plane: Uses Normal Equations (9x9) instead of Nx9 Matrix
    void extract_principal_plane_features_c(
        double* depth_data, int rows, int cols, double eps,
        double* out_features)
    {
        int n_pixels = rows * cols;
        double s1 = 0, s2 = 0, s3 = 0, s4 = 0; // For Z statistics
        
        // 9x9 Accumulators
        Eigen::Matrix<double, 9, 9> AtA = Eigen::Matrix<double, 9, 9>::Zero();
        Eigen::Matrix<double, 9, 1> Atb = Eigen::Matrix<double, 9, 1>::Zero();
        
        // Geometric Surface Area
        double As = 0.0;
        double Ap = (double)((cols - 1) * (rows - 1));

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                double z = depth_data[r * cols + c];

                // Stats Accumulation
                double z2 = z * z;
                s1 += z; s2 += z2; s3 += z2 * z; s4 += z2 * z2;

                // Plane Fitting Accumulation
                double x = (double)c;
                double y = (double)r;
                double p[9] = {1, x, y, x*x, x*y, y*y, x*x*y, x*y*y, y*y*y};

                for(int i=0; i<9; ++i) {
                    Atb(i) += p[i] * z;
                    for(int j=i; j<9; ++j) {
                        double val = p[i] * p[j];
                        AtA(i, j) += val;
                        if(i != j) AtA(j, i) += val;
                    }
                }

                // Surface Area (Rugosity)
                if (r < rows - 1 && c < cols - 1) {
                    double z1 = z;
                    double z2 = depth_data[(r + 1) * cols + c];
                    double z3 = depth_data[r * cols + (c + 1)];
                    double z4 = depth_data[(r + 1) * cols + (c + 1)];

                    double area1 = 0.5 * std::sqrt(std::pow(z3 - z1, 2) + std::pow(z2 - z1, 2) + 1.0);
                    double area2 = 0.5 * std::sqrt(std::pow(z4 - z2, 2) + std::pow(z4 - z3, 2) + 1.0);
                    As += area1 + area2;
                }
            }
        }

        // Solve System (9x9 is instant)
        Eigen::VectorXd coeffs = AtA.ldlt().solve(Atb);
        
        // Calculate Residuals (2nd Pass needed for exact residual stats)
        double res_s1 = 0, res_s2 = 0;
        
        // OpenMP helps here for larger patches
        #pragma omp parallel for reduction(+:res_s1, res_s2)
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                double z = depth_data[r * cols + c];
                double x = (double)c; 
                double y = (double)r;
                
                // Poly val
                double fit = coeffs(0) + coeffs(1)*x + coeffs(2)*y + 
                             coeffs(3)*x*x + coeffs(4)*x*y + coeffs(5)*y*y +
                             coeffs(6)*x*x*y + coeffs(7)*x*y*y + coeffs(8)*y*y*y;
                
                double diff = std::abs(z - fit);
                res_s1 += diff;
                res_s2 += diff * diff;
            }
        }

        double dist_mean = res_s1 / n_pixels;
        double dist_var = (res_s2 / n_pixels) - (dist_mean * dist_mean);
        double dist_std = std::sqrt(std::max(0.0, dist_var));

        // Normal Angle
        double n_x = coeffs(1);
        double n_y = coeffs(2);
        double n_z = -1.0; 
        double norm_mag = std::sqrt(n_x*n_x + n_y*n_y + n_z*n_z);
        double theta = std::acos(n_z / (norm_mag + eps)) * 180.0 / M_PI;

        double rugosity = (Ap > 0) ? As / Ap : 0.0;

        Stats z_st = calculate_stats_moments(s1, s2, s3, s4, n_pixels);

        int out_idx = 0;
        out_features[out_idx++] = z_st.std_dev; 
        out_features[out_idx++] = z_st.skew;
        out_features[out_idx++] = z_st.kurtosis;
        for(int i=0; i<9; ++i) out_features[out_idx++] = coeffs(i);
        out_features[out_idx++] = theta;
        out_features[out_idx++] = dist_mean;
        out_features[out_idx++] = dist_std;
        out_features[out_idx++] = rugosity;
    }

    // Optimized Curvatures: On-the-fly accumulation without allocating 6+8 vectors
    void extract_curvatures_c(
        double* depth_data, int rows, int cols, double eps,
        double* out_features)
    {
        // 8 metrics, 4 moments each (s1, s2) - we only need mean/std so s1, s2 suffice.
        double acc_s1[8] = {0};
        double acc_s2[8] = {0};

        // We need neighboring pixels. Boundary check inside loop is slow.
        // Loop from 1 to rows-1.
        
        #pragma omp parallel 
        {
            double loc_s1[8] = {0};
            double loc_s2[8] = {0};

            #pragma omp for nowait
            for(int r = 1; r < rows - 1; ++r) {
                // Pointers for 3 rows
                const double* r_up   = &depth_data[(r-1)*cols];
                const double* r_curr = &depth_data[r*cols];
                const double* r_down = &depth_data[(r+1)*cols];

                for(int c = 1; c < cols - 1; ++c) {
                    // Derivatives
                    double v_left  = r_curr[c-1];
                    double v_right = r_curr[c+1];
                    double v_up    = r_up[c];
                    double v_down  = r_down[c];
                    
                    double dx = 0.5 * (v_right - v_left);
                    double dy = 0.5 * (v_down - v_up);

                    // Second derivatives (central difference of 1st derivs)
                    // dxdx = (d/dx(x+1) - d/dx(x-1)) / 2 ... requires gathering neighbors' derivatives.
                    // This implies we DO need a buffer for 1st derivatives or re-compute them.
                    // To save memory, we re-compute neighbors. It's ALU heavy but Memory light.
                    // ALU is cheap. Cache misses are expensive.

                    // Neighbors for dxdx
                    double dx_r = 0.5 * (r_curr[c+2] - r_curr[c]); // Access c+2, careful boundary
                    double dx_l = 0.5 * (r_curr[c] - r_curr[c-2]);
                    // Actually, at boundary c=1, c-2 is -1. 
                    // To do this strictly 1-pass without buffer, we need a 3-line sliding window buffer for DX and DY.
                    // Given complexity, let's just compute basic DX/DY first into a buffer? 
                    // NO. Let's use the provided logic: we iterate r=1..rows-1.
                    // But we need dx at (r, c+1) and (r, c-1).
                    // Safe area is r=2..rows-2, c=2..cols-2.
                }
            }
        }
        
        // Fallback: The re-computation is complex to get right at boundaries.
        // Optimization: Use flat vectors but reuse them? 
        // Or just allocate 2 vectors (dx, dy) and compute dxdx from them.
        std::vector<double> dx(rows*cols);
        std::vector<double> dy(rows*cols);

        // Compute 1st derivs
        #pragma omp parallel for
        for(int r=0; r<rows; ++r) {
            for(int c=0; c<cols; ++c) {
                int idx = r*cols+c;
                dx[idx] = 0.5 * (depth_data[r*cols + clamp_idx(c+1, cols)] - depth_data[r*cols + clamp_idx(c-1, cols)]);
                dy[idx] = 0.5 * (depth_data[clamp_idx(r+1, rows)*cols + c] - depth_data[clamp_idx(r-1, rows)*cols + c]);
            }
        }

        // Compute 2nd derivs and stats
        #pragma omp parallel 
        {
            double loc_s1[8] = {0};
            double loc_s2[8] = {0};

            #pragma omp for nowait
            for(int i = 0; i < rows * cols; ++i) {
                int r = i / cols;
                int c = i % cols;

                double _dx = dx[i];
                double _dy = dy[i];

                double _dxdx = 0.5 * (dx[r*cols + clamp_idx(c+1, cols)] - dx[r*cols + clamp_idx(c-1, cols)]);
                double _dydy = 0.5 * (dy[clamp_idx(r+1, rows)*cols + c] - dy[clamp_idx(r-1, rows)*cols + c]);
                double _dxdy = 0.5 * (dx[clamp_idx(r+1, rows)*cols + c] - dx[clamp_idx(r-1, rows)*cols + c]);
                double _dydx = 0.5 * (dy[r*cols + clamp_idx(c+1, cols)] - dy[r*cols + clamp_idx(c-1, cols)]);

                double denom = 1.0 + _dx*_dx + _dy*_dy;
                double denom_sq = denom * denom;
                double G = (_dxdx * _dydy - _dxdy * _dydx) / denom_sq;
                double M = (_dydy + _dxdx) / (2.0 * std::pow(denom, 1.5));

                double discriminant = std::sqrt(std::max(M*M - G, 0.0));
                double k1 = M + discriminant;
                double k2 = M - discriminant;

                double S = (2.0 / M_PI) * std::atan2(k2 + k1, k2 - k1);
                double C = std::sqrt((k1*k1 + k2*k2) / 2.0);

                double nx = -_dx;
                double ny = -_dy;
                double nz = 1.0;
                double norm = std::sqrt(nx*nx + ny*ny + nz*nz) + eps;
                
                double alpha = std::atan2(ny/norm, nx/norm);
                double beta = std::atan2(nz/norm, std::sqrt(nx*nx + ny*ny)/norm);

                double vals[8] = {G, M, k1, k2, S, C, alpha, beta};

                for(int k=0; k<8; ++k) {
                    loc_s1[k] += vals[k];
                    loc_s2[k] += vals[k] * vals[k];
                }
            }
            
            #pragma omp critical
            {
                for(int k=0; k<8; ++k) {
                    acc_s1[k] += loc_s1[k];
                    acc_s2[k] += loc_s2[k];
                }
            }
        }

        int n = rows * cols;
        for(int i=0; i<8; ++i) {
            double mean = acc_s1[i] / n;
            double var = (acc_s2[i] / n) - (mean * mean);
            out_features[i*2] = mean;
            out_features[i*2+1] = std::sqrt(std::max(0.0, var));
        }
    }
}