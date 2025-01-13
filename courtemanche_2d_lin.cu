#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cuda_afib_utils.h"

// windows i/o
// #include <direct.h>

// linux i/o
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

//NOTE FOR TAU PARAMETERS: VALUES IN MANUSCRIPT TABLE 1 ARE EXPRESSED RELATIVE TO THE ABSOLUTE CAI VALUE
//IN THE CODE, WE USE NORMALIZED CALCIUM BY DIVIDING CAI BY 6e-4 (SEE CAI_NORM VARIABLE)
//THEREFORE, TO CONVERT FROM TABLE 1: TAU_K_NA = (9e-4 [table val] / (6e-4 [cai norm factor])) * 1e-6 (conversion mM - nM) = 1.5e6 (code value)


// parameters defined as macros - constants
#pragma region parameters
#define pi 3.141592f
#define Na_o 140.0f
#define K_o 5.4f
#define Ca_o 1.8f
#define R 8.3143f
#define T 310.0f
#define F 96.4867f
#define Cm 100.0f // pF

#define g_Na 7.8f
#define g_K1 0.09f
#define K_Q10 3.0f
#define g_to 0.1652f
#define g_Kr 0.029411765f
#define g_Ks 0.12941176f
#define g_Ca_L 0.12375f
#define Km_Na_i 10.0f
#define Km_K_o 1.5f
#define i_NaK_max 0.59933874f
#define g_B_Na 0.0006744375f
#define g_B_Ca 0.001131f
#define g_B_K 0.0f
#define I_NaCa_max 1600.0f
#define K_mNa 87.5f
#define K_mCa 1.38f
#define K_sat 0.1f
#define gamma 0.35f
#define i_CaP_max 0.275f
#define K_rel 30.0f
#define tau_tr 180.0f
#define I_up_max 0.005f
#define K_up 0.00092f
#define Ca_up_max 15.0f
#define CMDN_max 0.05f
#define TRPN_max 0.07f
#define CSQN_max 10.0f
#define Km_CMDN 0.00238f
#define Km_TRPN 0.0005f
#define Km_CSQN 0.8f
#define V_cell 20100.0f
#define V_i 1.3668e+04f
#define tau_f_Ca 2.0f
#define sigma 1.00091f
#define tau_u 8.00000f
#define V_rel 96.48f
#define V_up 1.10952e3f

#define Na_o_cube 2744e3f
#define Na_i 11.17f
#define Na_i_cube 1.393668613e03f
#define E_K -86.765281f
#define E_Na 67.541f

#define ecg_save_step 100 // every x ms/dt save ecg(sum) data

//variable conductances parameters
#define Ca_tgt_mean 0.43 // 0.43 - steady@1000
// 0.43-0.53 steady @ 1000 - 600
//0.38-0.48 - centered on steady@1000
#define Ca_tgt_lim1 0.33
#define Ca_tgt_lim2 0.53


#define tau_g 20e3                 //def 20/75
#define tau_g_x_scale (75.0*tau_g)

#define tau_k_D (0.135*tau_g_x_scale) //D goes to 0.0001 when kNa goes to 0.9
// #define tau_k_D (0.135*tau_g_x_scale) //D goes to 0.13 when kNa goes to 0.9
// #define tau_k_D (0.2*tau_g_x_scale) //D goes to 0.5 when kNa goes to 0.9
#define tau_g_D 1000.0 // slower timescale for D 1000 slow; 25 fast
#define tau_k_na (1.0*tau_g_x_scale)

// space and time parameters size - can do 1280 with 0.01dx
#define width 1024
#define height 1024

// Define the time step, diffusion coefficient, and grid spacing - hires
// #define dt 0.025f
// #define dx 0.00625f 
// #define dy 0.00625f

#define dt 0.1f
#define dx 0.0125f 
#define dy 0.0125f 


// spiral tip tracking definitions
#define tip_vecsize 5e7

#pragma endregion

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void courtemancheKernel
(float* mat_u,    float* mat_u_new, float* mat_u_norm,//mat_ = pointer to matrix
float* mat_Cai,  float* mat_Cai_new,
float* mat_Ca_rel,  float* mat_Ca_rel_new,
float* mat_Ca_up,  float* mat_Ca_up_new,

float* mat_m,  float* mat_m_new,
float* mat_h,  float* mat_h_new,
float* mat_j,  float* mat_j_new,

float* mat_oa,  float* mat_oa_new,
float* mat_oi,  float* mat_oi_new,
float* mat_ua,  float* mat_ua_new,
float* mat_ui,  float* mat_ui_new,

float* mat_xr,  float* mat_xr_new,
float* mat_xs,  float* mat_xs_new,
float* mat_d,  float* mat_d_new,
float* mat_f,  float* mat_f_new,

float* mat_f_Ca,  float* mat_f_Ca_new,
float* mat_uu,  float* mat_uu_new,
float* mat_v,  float* mat_v_new,
float* mat_w,  float* mat_w_new,

double* mat_Ca_tgt,
double* mat_k_D,      double* mat_k_D_new,    double* mat_m_k_D,    double* mat_m_k_D_new,
double* mat_k_na,     double* mat_k_na_new,   double* mat_m_k_na,   double* mat_m_k_na_new,

int t, float D, int Ca_feedback, int Ca_feedback_diff, int s1_spiral, int s2_spiral,
int periodic_boundaries, int spiral_flag, int opDiffSplit_flag, int pace_flag, int stim_pace, 
int* recent_tip_count, int* pace_beat_count, int pace_reentry_flag, int* pace_track, int* pace_track_count)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width && y < height) {

    float u      = mat_u[y * width + x];
    float Cai    = mat_Cai[y * width + x];
    float Ca_rel = mat_Ca_rel[y * width + x];
    float Ca_up  = mat_Ca_up[y * width + x];
    
    float m = mat_m[y * width + x];
    float h = mat_h[y * width + x];
    float j = mat_j[y * width + x];

    float oa = mat_oa[y * width + x];
    float oi = mat_oi[y * width + x];
    float ua = mat_ua[y * width + x];
    float ui = mat_ui[y * width + x];

    float xr = mat_xr[y * width + x];
    float xs = mat_xs[y * width + x];
    float d  = mat_d[y * width + x];
    float f  = mat_f[y * width + x];

    float f_Ca = mat_f_Ca[y * width + x];
    float uu   = mat_uu[y * width + x];
    float v    = mat_v[y * width + x];
    float w    = mat_w[y * width + x];

    double Ca_tgt = mat_Ca_tgt[y * width + x];
    double k_D    =  mat_k_D[y * width + x];

    double m_k_D =  mat_m_k_D[y * width + x];
    double k_na   =  mat_k_na[y * width + x];
    double m_k_na =  mat_m_k_na[y * width + x];

    double k_k1   =  -10.0 * (k_na) + 11.0;
    double k_ito  =  5.0 * (k_na) - 4.0;
    double k_kur  =  5.0 * (k_na) - 4.0;
    double k_ks   =  -10.0 * (k_na) + 11.0;
    
    double k_ca    =  5.0 * (k_na) - 4.0;
    double k_ryr   =  -10.0 * (k_na) + 11.0;
    double k_serca =  2.5 * (k_na) - 1.5;
    double k_naca  =  -4.0 * (k_na) + 5.0;

    float i_st = 0.0;

    // f_Ca gate
    float f_Cainfinity = 1.0f / (1.0f + Cai / 0.00035f);
    float diff_f_Ca = (f_Cainfinity - f_Ca) / tau_f_Ca;

    // f gate
    float f_infinity = expf(-(u + 28.0f) / 6.9f) / (1.0f + expf(-(u + 28.0f) / 6.9f));
    float tau_f = 9.0f * 1.0f / (0.0197f * expf(-0.00113569f * (u + 10.0f) * (u + 10.0f)) + 0.02f);
    float diff_f = (f_infinity - f) / tau_f;

    // d gate
    float d_infinity = 1.0f / (1.0f + expf((u + 10.0f) / -8.0f));
    float tau_d = (fabsf(u+10.0f)<0.1) ? 2.2894f : (1.0f - expf((u + 10.0f) / -6.240f)) / (0.0350f * (u + 10.0f) * (1.0f + expf((u + 10.0f) / -6.24f)));
    float diff_d = (d_infinity - d) / tau_d;

    // m gate
    float alpha_m = (fabsf(u + 47.13f) < 0.1) ? 3.2f : (0.32f * (u + 47.13f)) / (1.0f - 1.0f * expf(-0.1f * (u + 47.13f)));
    float beta_m = 0.08f * expf(-u / 11.0f);
    float m_inf = alpha_m / (alpha_m + beta_m);
    float tau_m = 1.0f / (alpha_m + beta_m);
    
    // h, j gates
    float aa = 1.0f - 1.0f / (1.0f + expf(-(u + 40.0f) / 0.24f));
    float alpha_h = aa * (0.135f * expf((u + 80.0f) / -6.8f));
    float beta_h = aa * (3.56f * expf(0.079f * u) + 310000.0f * expf(0.35f * u)) +
                   (1.0f - aa) * (1.0f / (0.13f * (1.0f + expf((u + 10.66f) / -11.1f))));
    float alpha_j = aa * (((-127140.0f * expf(0.24440f * u) - 3.474e-05 * expf(-0.04391f * u)) * (u + 37.78f)) 
                  / (1.0f + expf(0.311f * (u + 79.23f))));
    float beta_j = aa * ((0.1212f * expf(-0.01052f * u)) / (1.0f + expf(-0.1378f * (u + 40.14f)))) 
                 + (1.0f - aa) * ((0.3f * expf(-2.53500e-07f * u)) / (1.0f + expf(-0.1f * (u + 32.0f))));

    float h_inf = alpha_h / (alpha_h + beta_h);
    float tau_h = 1.0f / (alpha_h + beta_h);
    
    float j_inf = alpha_j / (alpha_j + beta_j);
    float tau_j = 1.0f / (alpha_j + beta_j);

    float diff_h = (h_inf - h) / tau_h;
    float diff_j = (j_inf - j) / tau_j;

    // oa, oi gates
    float alpha_oa = 0.65f * 1.0f / (expf((u - -10.0f) / -8.5f) + expf(((u - -10.0f) - 40.0f) / -59.0f));
    float beta_oa = 0.65f * 1.0f / (2.5f + expf(((u - -10.0f) + 72.0f) / 17.0f));
    float tau_oa = 1.0f / (alpha_oa + beta_oa) / K_Q10;
    float oa_infinity = 1.0f / (1.0f + expf(((u - -10.0f) + 10.47f) / -17.54f));
    float diff_oa = (oa_infinity - oa) / tau_oa;

    float alpha_oi = 1.0f / (18.53f + 1.0f * expf(((u - -10.0f) + 103.7f) / 10.95f));
    float beta_oi = 1.0f / (35.56f + 1.0f * expf(((u - -10.0f) - 8.74f) / -7.44f));
    float tau_oi = 1.0f / (alpha_oi + beta_oi) / K_Q10;
    float oi_infinity = 1.0f / (1.0f + expf(((u - -10.0f) + 33.1f) / 5.3f));
    float diff_oi = (oi_infinity - oi) / tau_oi;

    // ua, ui gates
    float alpha_ua = 0.65f * 1.0f / (expf((u - -10.0f) / -8.5f) + expf(((u - -10.0f) - 40.0f) / -59.0f));
    float beta_ua = 0.65f * 1.0f / (2.5f + expf(((u - -10.0f) + 72.0f) / 17.0f));
    float tau_ua = 1.0f / (alpha_ua + beta_ua) / K_Q10;
    float ua_infinity = 1.0f / (1.0f + expf(((u - -10.0f) + 20.3f) / -9.6f));
    float diff_ua = (ua_infinity - ua) / tau_ua;

    float alpha_ui = 1.0f / (21.0f + 1.0f * expf(((u - -10.0f) - 195.0f) / -28.0f));
    float beta_ui = 1.0f / expf(((u - -10.0f) - 168.0f) / -16.0f);
    float tau_ui = 1.0f / (alpha_ui + beta_ui) / K_Q10;
    float ui_infinity = 1.0f / (1.0f + expf(((u - -10.0f) - 109.45f) / 27.48f));
    float diff_ui = (ui_infinity - ui) / tau_ui;

    // xr, xs gates  
    float alpha_xr = (fabsf(u + 14.1f)<0.1) ? 0.0015f : (0.0003f * (u + 14.1f)) / (1.0f - expf((u + 14.1f) / -5.0f));
    float beta_xr = (fabsf(u - 3.3328f)<0.1) ? 3.7863e-04f : (7.38980e-05f * (u - 3.3328f)) / (expf((u - 3.3328f) / 5.1237f) - 1.0f);
    float tau_xr = 1.0f / (alpha_xr + beta_xr);
    float xr_infinity = 1.0f / (1.0f + expf((u + 14.1f) / -6.5f));
    float diff_xr = (xr_infinity - xr) / tau_xr;

    float alpha_xs = (fabsf(u-19.9f)<0.1) ? 6.8e-04f : (4.0e-05f * (u - 19.9f)) / (1.0f - expf((u - 19.9f) / -17.0f));
    float beta_xs = (fabsf(u-19.9f)<0.1) ? 3.15e-4f : (3.5e-05f * (u - 19.9f)) / (expf((u - 19.9f) / 9.0f) - 1.0f);
    float tau_xs = 0.50f * 1.0f / (alpha_xs + beta_xs);
    float xs_infinity = 1.0f / sqrt(1.0f + expf((u - 19.9f) / -12.7f));
    float diff_xs = (xs_infinity - xs) / tau_xs;

    // K+ currents
    float i_K1  = (k_k1) * (g_K1 * (u - E_K)) / (1.0f + expf(0.07f * (u + 80.0f)));
    float i_to  = k_ito * g_to * oa * oa * oa * oi * (u - E_K);
    float g_Kur = k_kur * 0.005f + 0.05f / (1.0f + expf((u - 15.0f) / -13.0f));
    float i_Kur = g_Kur * ua * ua * ua * ui * (u - E_K);
    float i_Kr  = (g_Kr * xr * (u - E_K)) / (1.0f + expf((u + 15.0f) / 22.4f));
    float i_Ks  = (k_ks) * g_Ks * xs * xs * (u - E_K);
    float f_NaK = 1.0f / (1.0f + 0.1245f * expf((-0.10f * F * u) / (R * T)) + 0.0365f * sigma * expf((-F * u) / (R * T)));
    float i_NaK = (((i_NaK_max * f_NaK * 1.0f) / (1.0f + 0.847f)) * K_o) / (K_o + Km_K_o);
    float i_B_K = g_B_K * (u - E_K);

    // Na+ currents
    float i_Na = k_na * g_Na * m * m * m * h * j * (u - E_Na);

    float i_NaCa = (k_naca) * (I_NaCa_max * (expf((gamma * F * u) / (R * T)) * Na_i_cube * Ca_o - expf(((gamma - 1.0f) * F * u) / (R * T)) * Na_o_cube * Cai)) 
                 / ((K_mNa * K_mNa * K_mNa + Na_o_cube) * (K_mCa + Ca_o) * (1.0f + K_sat * expf(((gamma - 1.0f) * u * F) / (R * T))));

    float i_B_Na = 1.0f * g_B_Na * (u - E_Na);

    // Ca2+ currents
    float i_Ca_L =  k_ca * g_Ca_L * d * f * f_Ca * (u - 65.0f);
    float i_CaP =  (i_CaP_max * Cai) / (0.0005f + Cai);
    float E_Ca = ((R * T) / (2.0f * F)) * logf(Ca_o / Cai);
    float i_B_Ca = g_B_Ca * (u - E_Ca);

    // Ca2+ handling
    float i_rel = K_rel * uu * uu * v * w * (Ca_rel - Cai); 
    float i_tr = (Ca_up - Ca_rel) / tau_tr;
    float diff_Ca_rel = (i_tr - i_rel) * 1.0f / (1.0f + (CSQN_max * Km_CSQN) / ((Ca_rel + Km_CSQN) * (Ca_rel + Km_CSQN)));
    float Fn = 1000.0f * (1.0e-15f * V_rel * i_rel - (1.0e-15f / (2.0f * F)) * (0.5f * Cm * i_Ca_L - 0.2f * Cm * i_NaCa));

    //ryr gates uu, v w
    float u_infinity = 1.0f / (1.0f + expf(-(Fn - 3.41750e-13f) / 1.36700e-15f));
    float diff_uu = (u_infinity - uu) / tau_u; //increase tau by 5 for slow pace alternans

    float tau_v = 1.91f + 2.09f * 1.0f / (1.0f + expf(-(Fn - 3.41750e-13f) / 1.36700e-15f));
    float v_infinity = 1.0f - 1.0f / (1.0f + expf(-(Fn - 6.83500e-14f) / 1.36700e-15f));
    float diff_v = (v_infinity - v) / tau_v;

    float tau_w = (fabsf(u - 7.9f)<0.1) ? 0.9231f :(6.0f * (1.0f - expf(-(u - 7.9f) / 5.0f))) / ((1.0f + 0.3f * expf(-(u - 7.9f) / 5.0f)) * (u - 7.9f));
    float w_infinity = 1.0f - 1.0f / (1.0f + expf(-(u - 40.0f) / 17.0f));
    float diff_w = (w_infinity - w) / tau_w;

    float i_up =  k_serca * I_up_max / (1.0f + K_up / Cai); 
    float i_up_leak = (k_ryr) * (I_up_max * Ca_up) / Ca_up_max;
    float diff_Ca_up = i_up - (i_up_leak + (i_tr * V_rel) / V_up);

    float B1 = Cm * (2.0f * i_NaCa - (i_CaP + i_Ca_L + i_B_Ca))/(2.0f * V_i * F) + (V_up * (i_up_leak - i_up) + i_rel * V_rel) / V_i;
    float B2 = 1.0f + (TRPN_max * Km_TRPN) / ((Cai + Km_TRPN) * (Cai + Km_TRPN)) + (CMDN_max * Km_CMDN) / ((Cai + Km_CMDN)*(Cai + Km_CMDN));
    float diff_Cai = B1 / B2;
 
    // spiral
    if (spiral_flag == 1) // (*recent_tip_count <= 5))
    {
        if ((t>s1_spiral) && (t<s1_spiral + 5.0/dt)){
            // if (x==0) i_st = -40.0f;
            if (x==0) u = 0.0f;
        }
        else if ((t>(s1_spiral + s2_spiral)) && (t<(s1_spiral+s2_spiral) + 5.0/dt)){
            // if ((x<width/2) && (y<height/2)) u = 20.0f;
            if ((x<width/2) && (y<height/2)) i_st = -20.0f;
        }
    }

    // pacemaker stim
    if (pace_flag == 1)
    {
        if ((t % stim_pace == 0) && (t > 0)) //&& (t*dt < 80e3))
        {

            if ((x * x + y * y) < (50 * 50)) u = 0.0f;
            if ((x == 2) && (y == 2))
            {
                pace_track[(*pace_track_count)++] = (int)(t * dt);
            }
            // if ((x*x + y*y) < 500) i_st = -180.0f;
            // if (x==0) u = (-20.0 + 88.0)/118.0;
            // printf("stim!\n");
        }
    }

        if ((pace_reentry_flag == 1) && (*recent_tip_count <= 5) && ((x * x + y * y) < (50 * 50)) )
    {
        if ((t % stim_pace == 0) && (t > 0))
        {    pace_beat_count[y * width + x] = 0;
            // if ((x == 2) && (y == 2)) printf("pace!")
        }
    }

    //15 beat after re-entry
    if ((pace_reentry_flag == 1) && (pace_beat_count[y * width + x] < 15) && ((x * x + y * y) < (50 * 50)))
    {
        if ((t % stim_pace == 0) && (t > 0))
        {
            u = 0.0f;
            pace_beat_count[y * width + x]++;
            if ((x == 2) && (y == 2))
                pace_track[(*pace_track_count)++] = (int)(t * dt);
        }
    }

    // if ((pace_reentry_flag == 1) && (*recent_tip_count <= 5))
    // {
    //     if ((t % stim_pace == 0) && (t > 0))
    //     {
    //         if ((x * x + y * y) < (50*50)) u = 0.0;
    //         if ((x == 2) && (y == 2))  
    //         {
    //             pace_track[(*pace_track_count)++] = (int)(t*dt); 
    //         }
    //     }
    // }

    float I_sum = i_Na + i_K1 + i_to + i_Kur + i_Kr + i_Ks + i_B_Na + i_B_K + i_B_Ca
          + i_NaK + i_CaP + i_NaCa + i_Ca_L + i_st;

    float diff_u = - I_sum; //Cm = 1

    if (isinf(diff_u)) {printf("Inf, %d, %d, %d, %f \n",x,y,t,i_Na); return;}
    if (isnan(diff_u)) {printf("Nan, %d, %d, %d, %f \n",x,y,t,u); return;}

    // if ((x==0) && (y==0)) printf(" x = %f ", Cai);
    // Ca feedback model (adapted from Marder)
    double dm_k_D = 0.0;        double dg_k_D = 0.0;
    double dm_k_na = 0.0;       double dg_k_na = 0.0;
    double Cai_norm = (double)Cai/6e-4;
    
    if (Ca_feedback == 1)  {
    // if ((Ca_feedback == 1) && (*recent_tip_count <= 5)) {
        dm_k_na = (Ca_tgt - Cai_norm)/tau_k_na;   
        dg_k_na = (m_k_na - k_na)/tau_g;
    }

    if (Ca_feedback_diff == 1) {
    // if ((Ca_feedback_diff == 1) && (*recent_tip_count <= 5)) {
        dm_k_D = (Ca_tgt - Cai_norm)/tau_k_D;     
        dg_k_D = (m_k_D - k_D)/(tau_g * tau_g_D);   
    }

    //debug
    // if ((x==0) && (y==0)) printf(" x = %f ", k_pos);
    // forward Euler integration

    mat_u_new[y * width + x]      = u + dt * diff_u;
    mat_u_norm[y * width + x] = (u +90.0f) / 100.0f;

    mat_Cai_new[y * width + x]    = Cai + dt * diff_Cai;
    mat_Ca_rel_new[y * width + x] = Ca_rel + dt * diff_Ca_rel;
    mat_Ca_up_new[y * width + x]  = Ca_up + dt * diff_Ca_up;

    // rush larsen for na gating
    mat_m_new[y * width + x] = m_inf + (m - m_inf) * expf(-dt/tau_m);

    // split na gating
    // for (int i_split = 0; i_split<10; i_split++)
    // {
    //     float diff_m = (m_inf - m) / tau_m;
    //     // float diff_h = (h_inf - h) / tau_h;
    //     // float diff_j = (j_inf - j) / tau_j;

    //     m = m + 0.01f*diff_m; //dt/10
    //     // h = h + 0.01f*diff_h;
    //     // j = j + 0.01f*diff_j;
    // }
    // mat_m_new[y * width + x] = m;

    //no split - req dt 0.01
    // float diff_m = (m_inf - m) / tau_m;
    // mat_m_new[y * width + x] = m + dt * diff_m;

    mat_h_new[y * width + x] = h + dt * diff_h;
    mat_j_new[y * width + x] = j + dt * diff_j;

    mat_oa_new[y * width + x] = oa + dt * diff_oa;
    mat_oi_new[y * width + x] = oi + dt * diff_oi;
    mat_ua_new[y * width + x] = ua + dt * diff_ua;
    mat_ui_new[y * width + x] = ui + dt * diff_ui;

    mat_xr_new[y * width + x] = xr + dt * diff_xr;
    mat_xs_new[y * width + x] = xs + dt * diff_xs;
    mat_d_new[y * width + x]  = d + dt * diff_d;
    mat_f_new[y * width + x]  = f + dt * diff_f;

    mat_f_Ca_new[y * width + x] = f_Ca + dt * diff_f_Ca;
    mat_uu_new[y * width + x]   = uu + dt * diff_uu;
    mat_v_new[y * width + x]    = v + dt * diff_v;
    mat_w_new[y * width + x]    = w + dt * diff_w;

    double k_d_new    = k_D    + (double)dt*dg_k_D;
    double m_k_D_new  = m_k_D  + (double)dt*dm_k_D;
    double k_na_new   = k_na   + (double)dt*dg_k_na;
    double m_k_na_new = m_k_na + (double)dt*dm_k_na;

    if (k_d_new <= 0.001)    k_d_new    = 0.001;    if (k_d_new    >= 2.0) k_d_new    = 2.0;
    if (m_k_D_new <= 0.001)  m_k_D_new  = 0.001;    if (m_k_D_new  >= 2.0) m_k_D_new  = 2.0;
    if (k_na_new <= 0.001)   k_na_new   = 0.001;    if (k_na_new   >= 2.0) k_na_new   = 2.0;
    if (m_k_na_new <= 0.001) m_k_na_new = 0.001;    if (m_k_na_new >= 2.0) m_k_na_new = 2.0;

    mat_k_D_new[y * width + x]    = k_d_new;
    mat_m_k_D_new[y * width + x]  = m_k_D_new;
    mat_k_na_new[y * width + x]   = k_na_new;
    mat_m_k_na_new[y * width + x] = m_k_na_new;

    //Cai gets weird with pacing sometimes
    if (mat_Cai_new[y * width + x] < 1e-8)    mat_Cai_new[y * width + x] =1e-8;
    if (mat_Ca_rel_new[y * width + x] < 1e-8) mat_Ca_rel_new[y * width + x] =1e-8;
    if (mat_Ca_up_new[y * width + x] < 1e-8)  mat_Ca_up_new[y * width + x] =1e-8;
    }
}

__global__ void diffusionKernelSharedMem
(float* mat_u, float* mat_u_new, double* mat_k_D, float D, float dt_diff, int periodic_boundaries)
{
    // grid, block, thread indexes
    int th_x = threadIdx.x;
    int th_y = threadIdx.y;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ float smem_D[8+2][8+2];  // bdim + 2*halo
    // __shared__ float smem_u[8+2][8+2];  // bdim + 2*halo

    if (x < width && y < height) 
    {
        // stencil indexes
        int ind_cy = y * width;
        int ind_cx = x;
        int ind_lx = (x > 0)         ? x - 1 : width - 1;
        int ind_rx = (x < width - 1) ? x + 1 : 0;
        int ind_ty = (y > 0)          ? (y-1) * width : (height - 1) * width;
        int ind_by = (y < height - 1) ? (y+1) * width : 0 * width;

        
        // load into shared mem
        // load D
        smem_D[th_y+1][th_x+1]                         = (float)mat_k_D[ind_cy + ind_cx];
        if (th_x==0)            smem_D[th_y+1][th_x]   = (float)mat_k_D[ind_cy + ind_lx]; //left edge
        if (th_x==blockDim.x-1) smem_D[th_y+1][th_x+2] = (float)mat_k_D[ind_cy + ind_rx]; //right edge
        if (th_y==0)            smem_D[th_y][th_x+1]   = (float)mat_k_D[ind_ty + ind_cx]; // top edge
        if (th_y==blockDim.y-1) smem_D[th_y+2][th_x+1] = (float)mat_k_D[ind_by + ind_cx]; // bottom edge

        if ((th_x==0) && (th_y==0))                       smem_D[th_y][th_x]       = (float)mat_k_D[ind_ty + ind_lx]; //left t corner
        if ((th_x==blockDim.x-1) && (th_y==0))            smem_D[th_y][th_x+2]     = (float)mat_k_D[ind_ty + ind_rx]; //right t corner
        if ((th_x==0) && (th_y==blockDim.y-1))            smem_D[th_y+2][th_x]     = (float)mat_k_D[ind_by + ind_lx]; //left b corner
        if ((th_x==blockDim.x-1) && (th_y==blockDim.y-1)) smem_D[th_y+2][th_x+2]   = (float)mat_k_D[ind_by + ind_rx]; //right b corner

        // // load u
        // smem_u[th_y+1][th_x+1] = mat_u[ind_cy + ind_cx];
        // if (th_x==0)        smem_u[th_y+1][th_x]   = mat_u[ind_cy + ind_lx]; //left edge
        // if (th_x==bdim_x-1) smem_u[th_y+1][th_x+2] = mat_u[ind_cy + ind_rx]; //right edge
        // if (th_y==0)        smem_u[th_y][th_x+1]   = mat_u[ind_ty + ind_cx]; // top edge
        // if (th_y==bdim_y-1) smem_u[th_y+2][th_x+1] = mat_u[ind_by + ind_cx]; // bottom edge

        // if ((th_x==0)        && (th_y==0))          smem_u[th_y][th_x]       = mat_u[ind_ty + ind_lx]; //left t corner
        // if ((th_x==bdim_x-1) && (th_y==0))          smem_u[th_y][th_x+2]     = mat_u[ind_ty + ind_rx]; //right t corner
        // if ((th_x==0)        && (th_y==bdim_y-1))   smem_u[th_y+2][th_x]     = mat_u[ind_by + ind_lx]; //left b corner
        // if ((th_x==bdim_x-1) && (th_y==bdim_y-1))   smem_u[th_y+2][th_x+2]   = mat_u[ind_by + ind_rx]; //right b corner

        __syncthreads(); // Synchronize to ensure all data is loaded


        //laplacian - impecabil... traiasca GPT
        float left, right, top, bottom, t_left, t_right, b_left, b_right, laplacian;
        // float D_center = (float)mat_k_D[y * width + x];
        float D_center = smem_D[th_y+1][th_x+1];
        float u        = mat_u[y * width + x];
        // float u = smem_u[th_y+1][th_x+1];
        

        //coefficients account for 9-point stencil - Oono-Puri 1988 (+ wiki)
        // can modify here to 1/0 instead of 0.5/0.25 for 5-point

        // 0.5*2 = 1 from 9pt 
        float D_l = (smem_D[th_y+1][th_x] * D_center)   / (smem_D[th_y+1][th_x] + D_center);
        float D_r = (smem_D[th_y+1][th_x+2] * D_center) / (smem_D[th_y+1][th_x+2] + D_center);
        float D_t = (smem_D[th_y][th_x+1] * D_center)   / (smem_D[th_y][th_x+1] + D_center);
        float D_b = (smem_D[th_y+2][th_x+1] * D_center) / (smem_D[th_y+2][th_x+1] + D_center);

        //0.25*2 = 0.5 from 9pt 
        float D_tl = 0.5f * (smem_D[th_y][th_x] * D_center)     / (smem_D[th_y][th_x] + D_center);
        float D_tr = 0.5f * (smem_D[th_y][th_x+2] * D_center)   / (smem_D[th_y][th_x+2] + D_center);
        float D_bl = 0.5f * (smem_D[th_y+2][th_x] * D_center)   / (smem_D[th_y+2][th_x] + D_center);
        float D_br = 0.5f * (smem_D[th_y+2][th_x+2] * D_center) / (smem_D[th_y+2][th_x+2] + D_center);

        // float D_l = ((float)mat_k_D[ind_cy + ind_lx] * D_center) / ((float)mat_k_D[ind_cy + ind_lx] + D_center);
        // float D_r = ((float)mat_k_D[ind_cy + ind_rx] * D_center) / ((float)mat_k_D[ind_cy + ind_rx] + D_center);
        // float D_t = ((float)mat_k_D[ind_ty + ind_cx] * D_center) / ((float)mat_k_D[ind_ty + ind_cx] + D_center);
        // float D_b = ((float)mat_k_D[ind_by + ind_cx] * D_center) / ((float)mat_k_D[ind_by + ind_cx] + D_center);

        // float D_tl = 0.5f * ((float)mat_k_D[ind_ty + ind_lx] * D_center) / ((float)mat_k_D[ind_ty + ind_lx] + D_center);
        // float D_tr = 0.5f * ((float)mat_k_D[ind_ty + ind_rx] * D_center) / ((float)mat_k_D[ind_ty + ind_rx] + D_center);
        // float D_bl = 0.5f * ((float)mat_k_D[ind_by + ind_lx] * D_center) / ((float)mat_k_D[ind_by + ind_lx] + D_center);
        // float D_br = 0.5f * ((float)mat_k_D[ind_by + ind_rx] * D_center) / ((float)mat_k_D[ind_by + ind_rx] + D_center);

        //at boundary: use values for periodic bd; in no flow, they just cancel out
        //replace lr with Dx and tb with Dy for anisotropic diff
        if (periodic_boundaries == 0)
        {
            left   = (x > 0)            ? D_l * mat_u[ind_cy + ind_lx] : D_l * u;  //mat_u * D_up
            right  = (x < width - 1)    ? D_r * mat_u[ind_cy + ind_rx] : D_r * u;
            top    = (y > 0)            ? D_t * mat_u[ind_ty + ind_cx] : D_t * u;
            bottom = (y < height - 1)   ? D_b * mat_u[ind_by + ind_cx] : D_b * u;

            t_left =  ((x > 0)          && (y > 0))          ? D_tl * mat_u[ind_ty + ind_lx] : D_tl * u;
            t_right = ((x < width - 1)  && (y > 0))          ? D_tr * mat_u[ind_ty + ind_rx] : D_tr * u;
            b_left =  ((x > 0)          && (y < height - 1)) ? D_bl * mat_u[ind_by + ind_lx] : D_bl * u;
            b_right = ((x < width - 1)  && (y < height - 1)) ? D_br * mat_u[ind_by + ind_rx] : D_br * u;
        }
        
        else
        {
            left   =  D_l * mat_u[ind_cy + ind_lx];  
            right  =  D_r * mat_u[ind_cy + ind_rx];
            top    =  D_t * mat_u[ind_ty + ind_cx];
            bottom =  D_b * mat_u[ind_by + ind_cx];

            t_left =   D_tl * mat_u[ind_ty + ind_lx];
            t_right =  D_tr * mat_u[ind_ty + ind_rx];
            b_left =   D_bl * mat_u[ind_by + ind_lx];
            b_right =  D_br * mat_u[ind_by + ind_rx];
        }

        laplacian = (D/ (dx * dx))*(left + right + top + bottom + t_left + t_right + b_left + b_right - u * (D_l + D_r + D_t + D_b + D_tl + D_tr + D_bl + D_br)); //dt at the end  

        mat_u_new[y * width + x]      = u + dt_diff * laplacian;
    }
}

__global__ void diffusionKernelMap(double* u, double D, double* u_new){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // double dx2_diff = 1.0;

    if (x < width && y < height) {
        double center = u[y * width + x];  //impecabil... traiasca GPT
        // double left = (x > 0) ? u[y * width + x - 1] : center;
        // double right = (x < width - 1) ? u[y * width + x + 1] : center;
        // double top = (y > 0) ? u[(y - 1) * width + x] : center;
        // double bottom = (y < height - 1) ? u[(y + 1) * width + x] : center;

        //periodic boundaries
        double left   = (x > 0)          ? u[y * width + x - 1]   : u[y * width + width - 1];
        double right  = (x < width - 1)  ? u[y * width + x + 1]   : u[y * width + 0];
        double top    = (y > 0)          ? u[(y - 1) * width + x] : u[(height - 1) * width + x] ;
        double bottom = (y < height - 1) ? u[(y + 1) * width + x] : u[(0) * width + x];

        // Apply no-flow boundary conditions
        double laplacian = (left + right + top + bottom - 4 * center) / 1.0;
        double newCenter = center + D * 1.0 * laplacian;
        u_new[y * width + x] = newCenter;
    }
}   

__global__ void tipTrackPhaseKernel(float* mat_u, float* mat_v, int *tip_count, int *recent_tip_count, long long int *tip_vec, int t)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x==1) && (y==1))
    {
        if (t % (int)(500 / dt) == 0) atomicAdd(recent_tip_count,-*recent_tip_count); // reset tip counter every x ms
        // printf("recent = %d \n", *recent_tip_count);
    }

    if ( (x>1) && (x<(width-2)) && (y>1) && (y<(height-2)) && ((x*x + y*y) > (75*75)) ) 
    {
        int i0   = ((y)   * width + (x));    //center
        int i1   = ((y-1) * width + (x));
        int i2   = ((y-1) * width + (x+1));
        int i3   = ((y)   * width + (x+1));
        int i4   = ((y+1) * width + (x+1));
        int i5   = ((y+1) * width + (x));
        int i6   = ((y+1) * width + (x-1));
        int i7   = ((y)   * width + (x-1));
        int i8   = ((y-1) * width + (x-1));

        //u is voltage, v is generic other value; Na h gating works
        float u0 = (mat_u[i0] + 90.0f) / 100.0f - 0.3f;
        float u1 = (mat_u[i1] + 90.0f) / 100.0f - 0.3f;
        float u2 = (mat_u[i2] + 90.0f) / 100.0f - 0.3f;
        float u3 = (mat_u[i3] + 90.0f) / 100.0f - 0.3f;
        float u4 = (mat_u[i4] + 90.0f) / 100.0f - 0.3f;
        float u5 = (mat_u[i5] + 90.0f) / 100.0f - 0.3f;
        float u6 = (mat_u[i6] + 90.0f) / 100.0f - 0.3f;
        float u7 = (mat_u[i7] + 90.0f) / 100.0f - 0.3f;
        float u8 = (mat_u[i8] + 90.0f) / 100.0f - 0.3f;

        float v0 = mat_v[i0] - 0.4f;
        float v1 = mat_v[i1] - 0.4f;
        float v2 = mat_v[i2] - 0.4f;
        float v3 = mat_v[i3] - 0.4f;
        float v4 = mat_v[i4] - 0.4f;
        float v5 = mat_v[i5] - 0.4f;
        float v6 = mat_v[i6] - 0.4f;
        float v7 = mat_v[i7] - 0.4f;
        float v8 = mat_v[i8] - 0.4f;

        float p0 = atan2f(u0, v0);
        float p1 = atan2f(u1, v1);
        float p2 = atan2f(u2, v2);
        float p3 = atan2f(u3, v3);
        float p4 = atan2f(u4, v4);
        float p5 = atan2f(u5, v5);
        float p6 = atan2f(u6, v6);
        float p7 = atan2f(u7, v7);
        float p8 = atan2f(u8, v8);

        int pint0, pint1, pint2, pint3, pint4, pint5, pint6, pint7, pint8;

        if (p0>0 && p0<pi/2) pint0 = 1;   if (p0>pi/2 && p0<pi) pint0 = 2;   if (p0>-pi && p0<-pi/2) pint0 = 3;   if (p0>-pi/2 && p0<0) pint0 = 4;
        if (p1>0 && p1<pi/2) pint1 = 1;   if (p1>pi/2 && p1<pi) pint1 = 2;   if (p1>-pi && p1<-pi/2) pint1 = 3;   if (p1>-pi/2 && p1<0) pint1 = 4;
        if (p2>0 && p2<pi/2) pint2 = 1;   if (p2>pi/2 && p2<pi) pint2 = 2;   if (p2>-pi && p2<-pi/2) pint2 = 3;   if (p2>-pi/2 && p2<0) pint2 = 4;
        if (p3>0 && p3<pi/2) pint3 = 1;   if (p3>pi/2 && p3<pi) pint3 = 2;   if (p3>-pi && p3<-pi/2) pint3 = 3;   if (p3>-pi/2 && p3<0) pint3 = 4;
        if (p4>0 && p4<pi/2) pint4 = 1;   if (p4>pi/2 && p4<pi) pint4 = 2;   if (p4>-pi && p4<-pi/2) pint4 = 3;   if (p4>-pi/2 && p4<0) pint4 = 4;
        if (p5>0 && p5<pi/2) pint5 = 1;   if (p5>pi/2 && p5<pi) pint5 = 2;   if (p5>-pi && p5<-pi/2) pint5 = 3;   if (p5>-pi/2 && p5<0) pint5 = 4;
        if (p6>0 && p6<pi/2) pint6 = 1;   if (p6>pi/2 && p6<pi) pint6 = 2;   if (p6>-pi && p6<-pi/2) pint6 = 3;   if (p6>-pi/2 && p6<0) pint6 = 4;
        if (p7>0 && p7<pi/2) pint7 = 1;   if (p7>pi/2 && p7<pi) pint7 = 2;   if (p7>-pi && p7<-pi/2) pint7 = 3;   if (p7>-pi/2 && p7<0) pint7 = 4;
        if (p8>0 && p8<pi/2) pint8 = 1;   if (p8>pi/2 && p8<pi) pint8 = 2;   if (p8>-pi && p8<-pi/2) pint8 = 3;   if (p8>-pi/2 && p8<0) pint8 = 4;

        int add1, add2, add3, add4, add5, add6, add7, add8 ;

        // if (pint0 == 1) add0 = 10;   if (pint0 == 2) add0 = 100;   if (pint0 == 3) add0 = 1000;   if (pint0 == 4) add0 = 10000;
        if (pint1 == 1) add1 = 10;   if (pint1 == 2) add1 = 100;   if (pint1 == 3) add1 = 1000;   if (pint1 == 4) add1 = 10000;
        if (pint2 == 1) add2 = 10;   if (pint2 == 2) add2 = 100;   if (pint2 == 3) add2 = 1000;   if (pint2 == 4) add2 = 10000;
        if (pint3 == 1) add3 = 10;   if (pint3 == 2) add3 = 100;   if (pint3 == 3) add3 = 1000;   if (pint3 == 4) add3 = 10000;
        if (pint4 == 1) add4 = 10;   if (pint4 == 2) add4 = 100;   if (pint4 == 3) add4 = 1000;   if (pint4 == 4) add4 = 10000;
        if (pint5 == 1) add5 = 10;   if (pint5 == 2) add5 = 100;   if (pint5 == 3) add5 = 1000;   if (pint5 == 4) add5 = 10000;
        if (pint6 == 1) add6 = 10;   if (pint6 == 2) add6 = 100;   if (pint6 == 3) add6 = 1000;   if (pint6 == 4) add6 = 10000;
        if (pint7 == 1) add7 = 10;   if (pint7 == 2) add7 = 100;   if (pint7 == 3) add7 = 1000;   if (pint7 == 4) add7 = 10000;
        if (pint8 == 1) add8 = 10;   if (pint8 == 2) add8 = 100;   if (pint8 == 3) add8 = 1000;   if (pint8 == 4) add8 = 10000;

        int lineint;
        lineint = pint0 + add1 + add2 + add3 + add4 + add5 + add6 + add7 + add8;

        //0 = self
        // int n0 = (lineint / 1) % 10;
        int n1 = (lineint / 10) % 10;
        int n2 = (lineint / 100) % 10;
        int n3 = (lineint / 1000) % 10;
        int n4 = (lineint / 10000) % 10;

        // ( (n0 == 1) & (n1<2) & (n2>1) & (n4>1)  )
        // (((n0 == 4) & (n2 >= 2)) || ((n0 == 1) & (n3 >= 1)) || ((n0 == 2) & (n1 >= 1) & (n2 >= 1) & (n3 >= 1) & (n4 >= 1)))
        // (((n0 == 4) & (n2 >= 2)))
        // ( (n0 == 1) & (n1<2) & (n2>1) & (n4>1)  )


        if( (n1 >= 1) & (n2 >= 1) & (n3 >= 1) & (n4 >= 1) )
        {
            // spiral_present
            atomicAdd(recent_tip_count,1); //recent tip count to decide if you pace or not

            int t_local = atomicAdd(tip_count, 3); //save tip coords
            tip_vec[t_local] = t; // time in ms
            tip_vec[t_local + 1] = x;
            tip_vec[t_local + 2] = y;
            // printf("%d, \n",tip_vec[*tip_count-1]);
        }
    }
}


int main(int argc, char *argv[] )
{

    // intialize arrays, memory
    #pragma region init arrays
    //initialize arrays on host (device memory alloc)
    float* h_u = new float[width * height]; //"new" is c++!
    float* h_u_norm = new float[width * height];
    float* h_Cai = new float[width * height]; 
    float* h_Ca_rel = new float[width * height]; 
    float* h_Ca_up = new float[width * height]; 

    float* h_m = new float[width * height]; 
    float* h_h = new float[width * height]; 
    float* h_j = new float[width * height]; 

    float* h_oa = new float[width * height]; 
    float* h_oi = new float[width * height]; 
    float* h_ua = new float[width * height]; 
    float* h_ui = new float[width * height]; 

    float* h_xr = new float[width * height]; 
    float* h_xs = new float[width * height]; 
    float* h_d = new float[width * height]; 
    float* h_f = new float[width * height]; 

    float* h_f_Ca = new float[width * height]; 
    float* h_uu = new float[width * height]; 
    float* h_v = new float[width * height]; 
    float* h_w = new float[width * height]; 

    double *h_Ca_tgt = new double[width * height];
    double* h_k_D = new double[width * height]; 
    double* h_m_k_D = new double[width * height]; 
    double* h_k_na = new double[width * height]; 
    double* h_m_k_na = new double[width * height]; 
    
    // initialize gpu arrays
    float* d_u; float* d_u_new; float* d_u_norm;
    float* d_Cai; float* d_Cai_new;
    float* d_Ca_rel; float* d_Ca_rel_new;
    float* d_Ca_up; float* d_Ca_up_new;

    float* d_m; float* d_m_new;
    float* d_h; float* d_h_new;
    float* d_j; float* d_j_new;

    float* d_oa; float* d_oa_new;
    float* d_oi; float* d_oi_new;
    float* d_ua; float* d_ua_new;
    float* d_ui; float* d_ui_new;

    float* d_xr; float* d_xr_new;
    float* d_xs; float* d_xs_new;
    float* d_d;  float* d_d_new;
    float* d_f;  float* d_f_new;

    float* d_f_Ca; float* d_f_Ca_new;
    float* d_uu;   float* d_uu_new;
    float* d_v;    float* d_v_new;
    float* d_w;    float* d_w_new;

    double *d_Ca_tgt; double *d_Ca_tgt_new;
    double* d_k_D;    double* d_k_D_new;    
    double* d_m_k_D;  double* d_m_k_D_new;
    double* d_k_na;   double* d_k_na_new; 
    double* d_m_k_na; double* d_m_k_na_new;

    //GPU memory allocation
    cudaMalloc(&d_u, sizeof(float) * width * height);      cudaMalloc(&d_u_new, sizeof(float) * width * height);
    cudaMalloc(&d_u_norm, sizeof(float) * width * height);
    cudaMalloc(&d_Cai, sizeof(float) * width * height);    cudaMalloc(&d_Cai_new, sizeof(float) * width * height);
    cudaMalloc(&d_Ca_rel, sizeof(float) * width * height); cudaMalloc(&d_Ca_rel_new, sizeof(float) * width * height);
    cudaMalloc(&d_Ca_up, sizeof(float) * width * height);  cudaMalloc(&d_Ca_up_new, sizeof(float) * width * height);

    cudaMalloc(&d_m, sizeof(float) * width * height);      cudaMalloc(&d_m_new, sizeof(float) * width * height);
    cudaMalloc(&d_h, sizeof(float) * width * height);      cudaMalloc(&d_h_new, sizeof(float) * width * height);
    cudaMalloc(&d_j, sizeof(float) * width * height);      cudaMalloc(&d_j_new, sizeof(float) * width * height);

    cudaMalloc(&d_oa, sizeof(float) * width * height);     cudaMalloc(&d_oa_new, sizeof(float) * width * height);
    cudaMalloc(&d_oi, sizeof(float) * width * height);     cudaMalloc(&d_oi_new, sizeof(float) * width * height);
    cudaMalloc(&d_ua, sizeof(float) * width * height);     cudaMalloc(&d_ua_new, sizeof(float) * width * height);
    cudaMalloc(&d_ui, sizeof(float) * width * height);     cudaMalloc(&d_ui_new, sizeof(float) * width * height);
 
    cudaMalloc(&d_xr, sizeof(float) * width * height);     cudaMalloc(&d_xr_new, sizeof(float) * width * height);
    cudaMalloc(&d_xs, sizeof(float) * width * height);     cudaMalloc(&d_xs_new, sizeof(float) * width * height);
    cudaMalloc(&d_d, sizeof(float) * width * height);      cudaMalloc(&d_d_new, sizeof(float) * width * height);
    cudaMalloc(&d_f, sizeof(float) * width * height);      cudaMalloc(&d_f_new, sizeof(float) * width * height);

    cudaMalloc(&d_f_Ca, sizeof(float) * width * height);   cudaMalloc(&d_f_Ca_new, sizeof(float) * width * height);
    cudaMalloc(&d_uu, sizeof(float) * width * height);     cudaMalloc(&d_uu_new, sizeof(float) * width * height);
    cudaMalloc(&d_v, sizeof(float) * width * height);      cudaMalloc(&d_v_new, sizeof(float) * width * height);
    cudaMalloc(&d_w, sizeof(float) * width * height);      cudaMalloc(&d_w_new, sizeof(float) * width * height);

    cudaMalloc(&d_Ca_tgt, sizeof(double) * width * height); cudaMalloc(&d_Ca_tgt_new, sizeof(double) * width * height);
    cudaMalloc(&d_k_D, sizeof(double) * width * height);    cudaMalloc(&d_k_D_new, sizeof(double) * width * height);
    cudaMalloc(&d_m_k_D, sizeof(double) * width * height);  cudaMalloc(&d_m_k_D_new, sizeof(double) * width * height);
    cudaMalloc(&d_k_na, sizeof(double) * width * height);   cudaMalloc(&d_k_na_new, sizeof(double) * width * height);
    cudaMalloc(&d_m_k_na, sizeof(double) * width * height); cudaMalloc(&d_m_k_na_new, sizeof(double) * width * height);
    #pragma endregion initalize 

    // initial conditions
    int load_data_flag = 0;
    int load_map_flag = 1;

    // choose save folder
    char load_file[256] = {0}; 
    sprintf(load_file, "saves/save_%s.csv", argv[1]); 
    // sprintf(load_file, "saves/save_5.csv");  //

    //choose map folder 
    char load_map_file_str[256] = {0}; 
    int map_smoothing = 10000;
    sprintf(load_map_file_str, "maps/1024x1024/%d/%s.csv", map_smoothing, argv[1]); //replace with actual input argument

    //default initial conditions
    if (load_data_flag == 0)  
    {
        // default
        initializeArrayValFloat(h_u, -81.18, width, height);
        initializeArrayValFloat(h_u_norm, 0.0, width, height);
        initializeArrayValFloat(h_Cai, 1.013e-4, width, height);
        initializeArrayValFloat(h_Ca_rel, 1.488, width, height);
        initializeArrayValFloat(h_Ca_up, 1.488, width, height);

        initializeArrayValFloat(h_m, 2.908e-3, width, height);
        initializeArrayValFloat(h_h, 9.649e-1, width, height);
        initializeArrayValFloat(h_j, 9.775e-1, width, height);

        initializeArrayValFloat(h_oa, 3.043e-2, width, height);
        initializeArrayValFloat(h_oi, 9.992e-1, width, height);
        initializeArrayValFloat(h_ua, 4.966e-3, width, height);
        initializeArrayValFloat(h_ui, 9.986e-1, width, height);

        initializeArrayValFloat(h_xr, 3.296e-5, width, height);
        initializeArrayValFloat(h_xs, 1.869e-2, width, height);
        initializeArrayValFloat(h_d, 1.367e-4, width, height);
        initializeArrayValFloat(h_f, 9.996e-1, width, height);

        initializeArrayValFloat(h_f_Ca, 7.755e-1, width, height);
        initializeArrayValFloat(h_uu, 0.00001, width, height);
        initializeArrayValFloat(h_v, 0.99, width, height);
        initializeArrayValFloat(h_w, 0.9992, width, height);

        initializeArrayValDouble(h_k_D, 1.0, width, height);
        initializeArrayValDouble(h_m_k_D, 1.0, width, height);
        initializeArrayValDouble(h_k_na, 1.0, width, height);
        initializeArrayValDouble(h_m_k_na, 1.0, width, height);
    }
    // load from file
    else if (load_data_flag == 1) 
    {
        FILE *input_file;
        double *data_in = (double *)malloc(sizeof(double) * width * height * 25);  
        input_file = fopen(load_file, "r");

        if (input_file == NULL)
        {
            printf("failed to open file %s \n", load_file);
            return 1;
        }

        for (int i_file = 0; i_file < width * height * 25; i_file++)
        {
            fscanf(input_file, "%lf", &data_in[i_file]);
            fscanf(input_file, ",");
        }

        fclose(input_file);

        for (int ix = 0; ix < width; ix++)
        {
            for (int iy = 0; iy < height; iy++)
            {
            h_u[ix * width + iy] = data_in[(width * height * 0) + ix * width + iy]; /// asta e
            h_u_norm[ix * width + iy] = data_in[(width * height * 1) + ix * width + iy];
            h_Cai[ix * width + iy] = data_in[(width * height * 2) + ix * width + iy];
            h_Ca_rel[ix * width + iy] = data_in[(width * height * 3) + ix * width + iy];
            h_Ca_up[ix * width + iy] = data_in[(width * height * 4) + ix * width + iy];

            h_m[ix * width + iy] = data_in[(width * height * 5) + ix * width + iy];
            h_h[ix * width + iy] = data_in[(width * height * 6) + ix * width + iy];
            h_j[ix * width + iy] = data_in[(width * height * 7) + ix * width + iy];

            h_oa[ix * width + iy] = data_in[(width * height * 8) + ix * width + iy];
            h_oi[ix * width + iy] = data_in[(width * height * 9) + ix * width + iy];
            h_ua[ix * width + iy] = data_in[(width * height * 10) + ix * width + iy];
            h_ui[ix * width + iy] = data_in[(width * height * 11) + ix * width + iy];

            h_xr[ix * width + iy] = data_in[(width * height * 12) + ix * width + iy];
            h_xs[ix * width + iy] = data_in[(width * height * 13) + ix * width + iy];
            h_d[ix * width + iy] = data_in[(width * height * 14) + ix * width + iy];
            h_f[ix * width + iy] = data_in[(width * height * 15) + ix * width + iy];

            h_f_Ca[ix * width + iy] = data_in[(width * height * 16) + ix * width + iy];
            h_uu[ix * width + iy] = data_in[(width * height * 17) + ix * width + iy];
            h_v[ix * width + iy] = data_in[(width * height * 18) + ix * width + iy];
            h_w[ix * width + iy] = data_in[(width * height * 19) + ix * width + iy];

            h_Ca_tgt[ix*width + iy] = data_in[(width*height*20) + ix*width + iy]; //ca_tgt is always copied from data if load
            h_k_D[ix * width + iy] = data_in[(width * height * 21) + ix * width + iy];
            h_m_k_D[ix * width + iy] = data_in[(width * height * 22) + ix * width + iy];
            h_k_na[ix * width + iy] = data_in[(width * height * 23) + ix * width + iy];
            h_m_k_na[ix * width + iy] = data_in[(width * height * 24) + ix * width + iy];
            }
        }
    }

    #pragma region reset ephys model - just for apd run
    // initializeArrayValFloat(h_u, -81.18, width, height);
    // initializeArrayValFloat(h_u_norm, 0.0, width, height);
    // initializeArrayValFloat(h_Cai, 1.013e-4, width, height);
    // initializeArrayValFloat(h_Ca_rel, 1.488, width, height);
    // initializeArrayValFloat(h_Ca_up, 1.488, width, height);

    // initializeArrayValFloat(h_m, 2.908e-3, width, height);
    // initializeArrayValFloat(h_h, 9.649e-1, width, height);
    // initializeArrayValFloat(h_j, 9.775e-1, width, height);

    // initializeArrayValFloat(h_oa, 3.043e-2, width, height);
    // initializeArrayValFloat(h_oi, 9.992e-1, width, height);
    // initializeArrayValFloat(h_ua, 4.966e-3, width, height);
    // initializeArrayValFloat(h_ui, 9.986e-1, width, height);

    // initializeArrayValFloat(h_xr, 3.296e-5, width, height);
    // initializeArrayValFloat(h_xs, 1.869e-2, width, height);
    // initializeArrayValFloat(h_d, 1.367e-4, width, height);
    // initializeArrayValFloat(h_f, 9.996e-1, width, height);

    // initializeArrayValFloat(h_f_Ca, 7.755e-1, width, height);
    // initializeArrayValFloat(h_uu, 0.00001, width, height);
    // initializeArrayValFloat(h_v, 0.99, width, height);
    // initializeArrayValFloat(h_w, 0.9992, width, height);
    #pragma endregion

    // Copy initial conditions to GPU
    #pragma region initial conditions
    cudaMemcpy(d_u, h_u, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_norm, h_u_norm, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cai, h_Cai, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ca_rel, h_Ca_rel, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ca_up, h_Ca_up, sizeof(float) * width * height, cudaMemcpyHostToDevice);

    cudaMemcpy(d_m, h_m, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h_h, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_j, h_j, sizeof(float) * width * height, cudaMemcpyHostToDevice);

    cudaMemcpy(d_oa, h_oa, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_oi, h_oi, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ua, h_ua, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ui, h_ui, sizeof(float) * width * height, cudaMemcpyHostToDevice);

    cudaMemcpy(d_xr, h_xr, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xs, h_xs, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, h_f, sizeof(float) * width * height, cudaMemcpyHostToDevice);

    cudaMemcpy(d_f_Ca, h_f_Ca, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_uu, h_uu, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, sizeof(float) * width * height, cudaMemcpyHostToDevice);

    cudaMemcpy(d_Ca_tgt, h_Ca_tgt, sizeof(double) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_D, h_k_D, sizeof(double) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m_k_D, h_m_k_D, sizeof(double) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_na, h_k_na, sizeof(double) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m_k_na, h_m_k_na, sizeof(double) * width * height, cudaMemcpyHostToDevice);
    #pragma endregion

    ///// Ca_tgt
    if ((load_data_flag == 0) && (load_map_flag == 0))    //generate Ca_tgt map
    {
        // srand(31); //random seed - def 5
        int rand_catgt_flag = 0;

        if (rand_catgt_flag == 1) initializeArrayValDoubleRandInt(h_Ca_tgt, 0.5, 0.5, width, height);
        if (rand_catgt_flag == 0) initializeArrayValDouble ( h_Ca_tgt, Ca_tgt_mean, width, height);
        
        cudaMemcpy(d_Ca_tgt, h_Ca_tgt, sizeof(double) * width * height, cudaMemcpyHostToDevice);
        // smooth Ca_tgt to create larger features
        if (rand_catgt_flag == 1)
        {
            // GPU kernel settings
            dim3 block_Catgt(8, 8);
            dim3 grid_Catgt((width + block_Catgt.x - 1) / block_Catgt.x, (height + block_Catgt.y - 1) / block_Catgt.y);

            for (int i_diff = 0; i_diff < 1500; i_diff += 2)
            {
            diffusionKernelMap<<<grid_Catgt, block_Catgt>>>(d_Ca_tgt, 0.2, d_Ca_tgt_new);
            diffusionKernelMap<<<grid_Catgt, block_Catgt>>>(d_Ca_tgt_new, 0.2, d_Ca_tgt);
            }

            cudaMemcpy(h_Ca_tgt, d_Ca_tgt, sizeof(double) * width * height, cudaMemcpyDeviceToHost);

            double max_catgt_sm = max2DArray(h_Ca_tgt, width, height);
            double min_catgt_sm = min2DArray(h_Ca_tgt, width, height);

            // amplify interval after smoothing
            for (int ix = 0; ix < width; ix++)
            {
            for (int iy = 0; iy < height; iy++)
            {
                h_Ca_tgt[ix * width + iy] = (h_Ca_tgt[ix * width + iy] - min_catgt_sm) / (max_catgt_sm - min_catgt_sm);
                h_Ca_tgt[ix * width + iy] = h_Ca_tgt[ix * width + iy] * (Ca_tgt_lim2 - Ca_tgt_lim1) + Ca_tgt_lim1;
            }
            }
            cudaMemcpy(d_Ca_tgt, h_Ca_tgt, sizeof(double) * width * height, cudaMemcpyHostToDevice);
        }
    }
    else if ((load_data_flag == 0) && (load_map_flag == 1)) 
    {
        FILE *input_map_file;
        double *data_in = (double *)malloc(sizeof(double) * width * height);
        input_map_file = fopen(load_map_file_str, "r");
        if (input_map_file == NULL)
        {
            printf("failed to open file %s \n", load_map_file_str);
            return 1;
        }

        for (int i_file = 0; i_file < width * height; i_file++)
        {
            fscanf(input_map_file, "%lf", &data_in[i_file]);
            fscanf(input_map_file, ",");
        }
        fclose(input_map_file);

        for (int ix = 0; ix < width; ix++)
        {
            for (int iy = 0; iy < height; iy++)
            {
                h_Ca_tgt[ix * width + iy] = data_in[(width * height * 0) + ix * width + iy];
                h_Ca_tgt[ix * width + iy] = h_Ca_tgt[ix * width + iy] * (Ca_tgt_lim2 - Ca_tgt_lim1) + Ca_tgt_lim1;
            }
        }
        cudaMemcpy(d_Ca_tgt, h_Ca_tgt, sizeof(double) * width * height, cudaMemcpyHostToDevice);
        
    }

    //time
    const long long int time = 25000*1000;// s * 1000 = ms
    const long long int numSteps = time/dt+10;   


    //diffusion coef
    float D = 1e-3;

    //diffusion time split
    int opDiffSplit_flag = 1;
    int numStepsDiff = 10;
    float dt_diff = dt/numStepsDiff;

    //flags
    int Ca_feedback = 1;
    int Ca_feedback_diff = 1;
    int periodic_boundaries = 0;
    
    int state_save_flag = 1;
    int data_save_flag = 1;

    int just_pace_flag = 0;     //just pace in the corner
    int pace_n_sp_flag = 0;     //pace followed by spiral
    int pace_reentry_flag = 1;  //pace and check for reentry

    int spiral_flag_kern = 0;
    int pace_flag_kern = 0; //always 0 - gets set to 1 with appropriate flag
    int spiral_track_flag = 1;
    
    //spiral track parameters
    int spiral_track_step = 10;
    int count_tip_tracks = 0;
    int count_tip_save = 1;

    //stim times
    int s1_spiral = 0.1e3/dt;
    int s2_spiral = 350/dt;

    //pause at the start to remove initial cond  
    int stim_time_1 = 1000/dt;  
    // int stim_time_2 = 750/dt;   
    // int stim_time_3 = 500/dt;
    // int stim_time_4 = 400/dt;
    // int stim_time_5 = 350/dt;
    // int stim_time_6 = 300/dt;
    // int stim_time_7 = 250/dt;
    // int stim_time_8 = 200/dt;
    // int stim_time_9 = 150/dt;
    // int stim_time_10 = 100/dt;

    int stim_pace = 100/dt;  

    //pace and stop parameters
    int pace_on_counter = 0;
    int pace_on_limit = 10*1000/dt; //10 def
    int pace_off_counter = 0;
    int pace_off_limit = 1*100/dt;
    int pace_sp_counter = 0;

    int b1_spiral = 0;
    int b2_spiral = 0;
    int b3_spiral = 0;

    //save params: times, folders
    int k_save_data = 1; //save counter
    int k_save_state = 1; //save counter

    const int data_save_interval = 100*1000/dt; //steps (ms/dt) def 100*1000 zsave 1*10 apd calc 1
    const int state_save_interval = 50000*1000/dt; //steps (ms/dt)

    //save folder
    char foldname[256] = {0}; 
    char foldname_save[256] = {0}; 
    char foldname_track[256] = {0}; 

    char savefile[256] = {0}; 
    char savename[256] = {0}; 
    char save_folder_name[256] = {0};
    
    //name of folder where we save data
    // sprintf(save_folder_name, "pace_reentry_1024_c225lfeat_nd225_1");   
    // sprintf(save_folder_name, "longerD_l1_1024_2075_c100_sm_%d_map_%s", map_smoothing, argv[1]);  
 
    sprintf(save_folder_name, "1024_2075_15beat_tgint_9pt_125_sm_%d_map_%s", map_smoothing, argv[1]);
    // sprintf(save_folder_name, "1024_2075_jpace_tgint_9pt_sm_%d_map_%s", map_smoothing, argv[1]);
    // sprintf(save_folder_name, "1024_2075_D25em3_sm_%d_map_%s", map_smoothing, argv[1]);
    // sprintf(save_folder_name, "512_fin_2075_D1e3_c100_pacensp_043_9pt_125");
    // sprintf(save_folder_name, "1024_2075_D1000_pacensp_043");
    // sprintf(save_folder_name, "map8_end_apd");


    sprintf(foldname, "../cuda_csv/");
    strcat(foldname, save_folder_name);
    strcat(foldname, "/");

    strcpy(foldname_save, foldname);
    strcat(foldname_save, "save/");

    strcpy(foldname_track, foldname);
    strcat(foldname_track, "tiptrack/");

    // windows folders
    // _mkdir(foldname);
    // _mkdir(foldname_save);
    // _mkdir(foldname_track);

    // linux folders
    mkdir(foldname, 0777);
    mkdir(foldname_save, 0777);
    mkdir(foldname_track, 0777);

    // save Ca_tgt to folder
    sprintf(savefile, "Ca_tgt.csv");
    strcpy(savename, foldname);
    strcat(savename, savefile);
    writeMatrixCsvDouble(h_Ca_tgt, savename, 0, width, height);

    // initialize ecg, tip track matrix
    #pragma region
    //initialize pseudo-ecg matrix
    int ecg_save_size = numSteps/ecg_save_step;
    float* h_ecg = new float[ecg_save_size]; //"new" is c++!
    initializeArrayValFloat1D(h_ecg, 0.0f,ecg_save_size);
    float ecg_cublas_res = 0.0;
    float ecg_cublas_res_scaled = 0.0;
    int ecg_counter = 0;

    //pace track matrix + remember to copy utils too
    //ecg_save_size is actuall ~100 times larger than this,even for the shortest cycle, but we have memory and its ok
    int *h_pace_track = new int[ecg_save_size]; 
    initializeVecValInt(h_pace_track, 0, ecg_save_size);
    int *d_pace_track;
    cudaMalloc(&d_pace_track, ecg_save_size * sizeof(int));
    cudaMemcpy(d_pace_track, h_pace_track, ecg_save_size *sizeof(int), cudaMemcpyHostToDevice);

    int *d_pace_track_count;
    int h_pace_track_count = 0;
    cudaMalloc(&d_pace_track_count, sizeof(int));
    cudaMemcpy(d_pace_track_count, &h_pace_track_count, sizeof(int), cudaMemcpyHostToDevice);

    // spiral tip data
    int *d_tip_count;
    int h_tip_count = 0;
    cudaMalloc(&d_tip_count, sizeof(int));
    cudaMemcpy(d_tip_count, &h_tip_count, sizeof(int), cudaMemcpyHostToDevice);

    int *d_recent_tip_count;
    int h_recent_tip_count = 0;
    cudaMalloc(&d_recent_tip_count, sizeof(int));
    cudaMemcpy(d_recent_tip_count, &h_recent_tip_count, sizeof(int), cudaMemcpyHostToDevice);

    //initialize with high value so it doesn't pace at start of sim
    //pace beat counter - paceds for no of beats after re-entry
    int* h_pace_beat_count = new int[width * height];
    initializeArrayValInt(h_pace_beat_count, 1000, width, height); 
    int* d_pace_beat_count;  
    cudaMalloc(&d_pace_beat_count, sizeof(int) * width * height); 
    cudaMemcpy(d_pace_beat_count, h_pace_beat_count, sizeof(int) * width * height, cudaMemcpyHostToDevice);


    long long int *h_tip_vec = (long long int *)malloc(tip_vecsize * sizeof(long long int));
    initializeVecValLongLongInt(h_tip_vec, 0, tip_vecsize);

    long long int *d_tip_vec;
    cudaMalloc(&d_tip_vec, tip_vecsize * sizeof(long long int));
    cudaMemcpy(d_tip_vec, h_tip_vec, tip_vecsize * sizeof(long long int), cudaMemcpyHostToDevice);
    #pragma endregion
    
    // GPU kernel settings
    dim3 block(8, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    //main for
    clock_t start = clock();
    for (int t = 0; t < numSteps; t+=2){
        // if (t*dt>135e3) 
        // {
        //     Ca_feedback = 0;
        //     Ca_feedback_diff = 1;
        // }

        // save detailed data in intervals  
        // if ((t * dt >= 893e3) && (t * dt <= 898e3))
        // {
        //     data_save_flag = 1;
        // }
        // else if ((t * dt >= 3000e3) && (t * dt <= 3005e3))
        // {
        //     data_save_flag = 1;
        // }
        // else data_save_flag = 0;

        courtemancheKernel<<<grid, block>>>
        (   d_u, d_u_new, d_u_norm,
            d_Cai,  d_Cai_new,
            d_Ca_rel,  d_Ca_rel_new,
            d_Ca_up,  d_Ca_up_new,

            d_m,  d_m_new,
            d_h,  d_h_new,
            d_j,  d_j_new,

            d_oa,  d_oa_new,
            d_oi,  d_oi_new,
            d_ua,  d_ua_new,
            d_ui,  d_ui_new,

            d_xr,  d_xr_new,
            d_xs,  d_xs_new,
            d_d,  d_d_new,
            d_f,  d_f_new,

            d_f_Ca,  d_f_Ca_new,
            d_uu,  d_uu_new,
            d_v,  d_v_new,
            d_w,  d_w_new,

            d_Ca_tgt,
            d_k_D,     d_k_D_new,     d_m_k_D,     d_m_k_D_new,
            d_k_na,    d_k_na_new,    d_m_k_na,    d_m_k_na_new,
            
            t,D, Ca_feedback, Ca_feedback_diff, s1_spiral, s2_spiral,
            periodic_boundaries, spiral_flag_kern, opDiffSplit_flag, pace_flag_kern, stim_pace, 
            d_recent_tip_count, d_pace_beat_count, pace_reentry_flag, d_pace_track, d_pace_track_count                        
        );
        gpuErrchk( cudaPeekAtLastError() );

        if (opDiffSplit_flag == 1){
            for (int t_diff = 0; t_diff<numStepsDiff; t_diff+=2)
            {
                diffusionKernelSharedMem<<<grid, block>>>
                (d_u_new, d_u, d_k_D,  D,  dt_diff, periodic_boundaries);
                diffusionKernelSharedMem<<<grid, block>>>
                (d_u, d_u_new, d_k_D,  D,  dt_diff, periodic_boundaries);
            }
        }
        else
        {
            diffusionKernelSharedMem<<<grid, block>>>
            (d_u_new, d_u, d_k_D,  D,  dt, periodic_boundaries);
            d_u_new = d_u;
        }

        gpuErrchk( cudaPeekAtLastError() );
        
        courtemancheKernel<<<grid, block>>>
        (   d_u_new,    d_u, d_u_norm,
            d_Cai_new,  d_Cai,
            d_Ca_rel_new,  d_Ca_rel,
            d_Ca_up_new,  d_Ca_up,

            d_m_new,  d_m,
            d_h_new,  d_h,
            d_j_new,  d_j,

            d_oa_new,  d_oa,
            d_oi_new,  d_oi,
            d_ua_new,  d_ua,
            d_ui_new,  d_ui,

            d_xr_new,  d_xr,
            d_xs_new,  d_xs,
            d_d_new,  d_d,
            d_f_new,  d_f,

            d_f_Ca_new,  d_f_Ca,
            d_uu_new,  d_uu,
            d_v_new,  d_v,
            d_w_new,  d_w,

            d_Ca_tgt,
            d_k_D_new,     d_k_D,     d_m_k_D_new,     d_m_k_D,
            d_k_na_new,    d_k_na,    d_m_k_na_new,    d_m_k_na,
            
            t+1,D, Ca_feedback, Ca_feedback_diff, s1_spiral, s2_spiral,
            periodic_boundaries, spiral_flag_kern, opDiffSplit_flag, pace_flag_kern, stim_pace, 
            d_recent_tip_count, d_pace_beat_count, pace_reentry_flag, d_pace_track, d_pace_track_count
        );
        gpuErrchk( cudaPeekAtLastError() );
        
        if (opDiffSplit_flag == 1){
        for (int t_diff = 0; t_diff<numStepsDiff; t_diff+=2)
            {
                diffusionKernelSharedMem<<<grid, block>>>
                (d_u, d_u_new, d_k_D_new,  D,  dt_diff, periodic_boundaries);
                diffusionKernelSharedMem<<<grid, block>>>
                (d_u_new, d_u, d_k_D_new,  D,  dt_diff, periodic_boundaries);
            }
        }
        else     
        {
            diffusionKernelSharedMem<<<grid, block>>>
            (d_u, d_u_new, d_k_D_new,  D,  dt, periodic_boundaries);
            d_u = d_u_new;
        }


        if (t%ecg_save_step == 0)
        {
            cublasHandle_t handle;
            cublasStatus_t ret = cublasCreate(&handle);
            ret = cublasSasum(handle, width * height, d_u_norm, 1, &ecg_cublas_res);
            ecg_cublas_res_scaled = ecg_cublas_res/(width*height);
            h_ecg[ecg_counter++] = ecg_cublas_res_scaled;
            cublasDestroy(handle);
        }

        //just pace
        if (just_pace_flag == 1)
        {
            pace_flag_kern = 1;
            stim_pace = stim_time_1;
            // if ((t*dt >= 0e3)    && (t*dt <= 2000e3)) stim_pace = stim_time_1;
            // if ((t*dt >= 2000e3) && (t*dt <= 5000e3)) stim_pace = stim_time_2;
            // if ((t*dt >= 0e3)  && (t*dt <= 10e3)) stim_pace = stim_time_1;
            // if ((t*dt >= 10e3) && (t*dt <= 20e3)) stim_pace = stim_time_2;
            // if ((t*dt >= 20e3) && (t*dt <= 30e3)) stim_pace = stim_time_3;
            // if ((t*dt >= 30e3) && (t*dt <= 40e3)) stim_pace = stim_time_4;
            // if ((t*dt >= 40e3) && (t*dt <= 50e3)) stim_pace = stim_time_5;
            // if ((t*dt >= 50e3) && (t*dt <= 60e3)) stim_pace = stim_time_6;
            // if ((t*dt >= 60e3) && (t*dt <= 70e3)) stim_pace = stim_time_7;
            // if ((t*dt >= 70e3) && (t*dt <= 80e3)) stim_pace = stim_time_8;
            // if ((t*dt >= 80e3) && (t*dt <= 90e3)) stim_pace = stim_time_9;
            // if ((t*dt >= 90e3) && (t*dt <= 100e3)) stim_pace = stim_time_10;
        }

        //pace with cross-stim
        if (pace_n_sp_flag == 1)
        {
            if ((pace_on_counter < pace_on_limit) )
            {
                pace_flag_kern = 1;
                pace_on_counter+=2;
                // printf("counter = %d \n", pace_on_counter);
            }
            else if (pace_off_counter < pace_off_limit)
            {
                pace_flag_kern = 0;
                pace_off_counter +=2;
                // printf("counter2 = %d \n", pace_off_counter);
            }
            else if ((b1_spiral == 1) && (ecg_cublas_res_scaled<0.1))
            {
                spiral_flag_kern = 1;
                s1_spiral = t;
                // s2_spiral = 350/dt; //D25e4, 
                // s2_spiral = 325/dt;    //D1e3 0.53
                s2_spiral = 225/dt;    //D1e3 0.33, 0.43
                b1_spiral = 0;
                pace_off_counter = 0;
                printf("1!!!! \n");
            }
            else if ((b2_spiral == 1) && (ecg_cublas_res_scaled<0.1))
            {
                spiral_flag_kern = 1;
                s1_spiral = t;
                // s2_spiral = 325/dt;
                // s2_spiral = 300/dt;
                s2_spiral = 200/dt;
                b2_spiral = 0;
                pace_off_counter = 0;
                printf("2!!!! \n");
            }
            else if ((b3_spiral == 1) && (ecg_cublas_res_scaled<0.1))
            {
                spiral_flag_kern = 1;
                s1_spiral = t;
                // s2_spiral = 300/dt;
                // s2_spiral = 275/dt;
                s2_spiral = 175/dt; 
                b3_spiral = 0;
                pace_off_counter = 0;
                printf("3!!!! \n");
            }

            else if ((pace_off_counter >= pace_off_limit) && (ecg_cublas_res_scaled<0.1))
            {
                pace_on_counter = 0;
                pace_off_counter = 0;
                b1_spiral = 1;
                b2_spiral = 1;
                b3_spiral = 1;
                printf("reset \n");
            }

            if (((b1_spiral ==0) || (b2_spiral == 0)) && b3_spiral == 1)
            {
                pace_sp_counter +=2;
            }

            if (pace_sp_counter >= pace_on_limit )
            {
                pace_sp_counter = 0;
                b1_spiral = 0;
                b2_spiral = 0;
                b3_spiral = 0;
            }

        }

        // spiral track
        gpuErrchk(cudaPeekAtLastError());
        if ((spiral_track_flag == 1) && (t % spiral_track_step == 0) && (t>0))
        {
            tipTrackPhaseKernel<<<grid, block>>>(d_u, d_h, d_tip_count, d_recent_tip_count, d_tip_vec, t);
            gpuErrchk(cudaPeekAtLastError());
            count_tip_tracks++;
        }

        if (count_tip_tracks > (tip_vecsize/(3*100)))   //if number of tip data points gets close to the vecsize; 3 coordinates, at most 100 spiral tips
        {
            sprintf(savefile, "tiptrack_%d.csv", count_tip_save++);
            strcpy(savename, foldname_track);
            strcat(savename, savefile);
            printf("writing file: %s\n", savename); 

            cudaMemcpy(h_tip_vec, d_tip_vec, tip_vecsize*sizeof(long long int), cudaMemcpyDeviceToHost);
            writeVecCsvLongLongInt(h_tip_vec, savename, 0, tip_vecsize);   

            initializeVecValLongLongInt(h_tip_vec, 0, tip_vecsize); //reset data
            cudaMemcpy(d_tip_vec, h_tip_vec, tip_vecsize * sizeof(long long int), cudaMemcpyHostToDevice);

            cudaMemcpy(d_tip_count, &h_tip_count, sizeof(int), cudaMemcpyHostToDevice); // reset device data counter
            count_tip_tracks = 0;
            gpuErrchk(cudaPeekAtLastError());
        }


        // save data: every x steps copy data to host and write to device
        if ((data_save_flag == 1) && (t % data_save_interval == 0) && (t > 0))
        {
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            cudaMemcpy(h_u, d_u, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_h, d_h, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_Cai, d_Cai, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_k_na, d_k_na, sizeof(double) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_k_D, d_k_D, sizeof(double) * width * height, cudaMemcpyDeviceToHost);

            sprintf(savefile, "data_%d.csv", k_save_data++);
            strcpy(savename, foldname);
            strcat(savename, savefile);

            int append_val = 0;
            writeMatrixCsvFloat(h_u, savename, append_val++, width, height);
            writeMatrixCsvFloat(h_h, savename, append_val++, width, height);
            writeMatrixCsvFloat(h_Cai, savename, append_val++, width, height);
            writeMatrixCsvDouble(h_k_na, savename, append_val++, width, height);
            writeMatrixCsvDouble(h_k_D, savename, append_val++, width, height);
            printf("writing file: %s\n", savename);

            sprintf(savefile, "ecg.csv");
            strcpy(savename, foldname);
            strcat(savename, savefile);
            printf("writing file: %s\n", savename);
            writeVecCsvFloat(h_ecg, savename, 0, ecg_save_size);

            cudaMemcpy(h_pace_track, d_pace_track, ecg_save_size *sizeof(int), cudaMemcpyDeviceToHost);
            sprintf(savefile, "pace_track.csv");
            strcpy(savename, foldname);
            strcat(savename, savefile);
            printf("writing file: %s\n", savename);
            writeVecCsvInt(h_pace_track, savename, 0, ecg_save_size);
        }

        // save state
        if (((t % state_save_interval == 0) && (t > 0) && (state_save_flag == 1)) || (t == numSteps - 2))
        {
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            cudaMemcpy(h_u, d_u, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_u_norm, d_u_norm, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_Cai, d_Cai, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_Ca_rel, d_Ca_rel, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_Ca_up, d_Ca_up, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

            cudaMemcpy(h_m, d_m, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_h, d_h, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_j, d_j, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

            cudaMemcpy(h_oa, d_oa, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_oi, d_oi, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ua, d_ua, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ui, d_ui, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

            cudaMemcpy(h_xr, d_xr, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_xs, d_xs, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_d, d_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_f, d_f, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

            cudaMemcpy(h_f_Ca, d_f_Ca, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_uu, d_uu, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_v, d_v, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_w, d_w, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

            cudaMemcpy(h_Ca_tgt, d_Ca_tgt, sizeof(double) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_k_D, d_k_D, sizeof(double) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_m_k_D, d_m_k_D, sizeof(double) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_k_na, d_k_na, sizeof(double) * width * height, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_m_k_na, d_m_k_na, sizeof(double) * width * height, cudaMemcpyDeviceToHost);

            sprintf(savefile, "save_%d.csv", k_save_state++);
            strcpy(savename, foldname_save);
            strcat(savename, savefile);

            int append_val = 0;

            writeMatrixCsvFloatStateSave(h_u, savename, append_val++, width, height);
            writeMatrixCsvFloatStateSave(h_u_norm, savename, append_val++, width, height);
            writeMatrixCsvFloatStateSave(h_Cai, savename, append_val++, width, height);
            writeMatrixCsvFloatStateSave(h_Ca_rel, savename, append_val++, width, height);
            writeMatrixCsvFloatStateSave(h_Ca_up, savename, append_val++, width, height);

            writeMatrixCsvFloatStateSave(h_m, savename, append_val++, width, height);
            writeMatrixCsvFloatStateSave(h_h, savename, append_val++, width, height);
            writeMatrixCsvFloatStateSave(h_j, savename, append_val++, width, height);

            writeMatrixCsvFloatStateSave(h_oa, savename, append_val++, width, height);
            writeMatrixCsvFloatStateSave(h_oi, savename, append_val++, width, height);
            writeMatrixCsvFloatStateSave(h_ua, savename, append_val++, width, height);
            writeMatrixCsvFloatStateSave(h_ui, savename, append_val++, width, height);

            writeMatrixCsvFloatStateSave(h_xr, savename, append_val++, width, height);
            writeMatrixCsvFloatStateSave(h_xs, savename, append_val++, width, height);
            writeMatrixCsvFloatStateSave(h_d, savename, append_val++, width, height);
            writeMatrixCsvFloatStateSave(h_f, savename, append_val++, width, height);

            writeMatrixCsvFloatStateSave(h_f_Ca, savename, append_val++, width, height);
            writeMatrixCsvFloatStateSave(h_uu, savename, append_val++, width, height);
            writeMatrixCsvFloatStateSave(h_v, savename, append_val++, width, height);
            writeMatrixCsvFloatStateSave(h_w, savename, append_val++, width, height);

            writeMatrixCsvDoubleStateSave(h_Ca_tgt, savename, append_val++, width, height);
            writeMatrixCsvDoubleStateSave(h_k_D, savename, append_val++, width, height);
            writeMatrixCsvDoubleStateSave(h_m_k_D, savename, append_val++, width, height);
            writeMatrixCsvDoubleStateSave(h_k_na, savename, append_val++, width, height);
            writeMatrixCsvDoubleStateSave(h_m_k_na, savename, append_val++, width, height);

            printf("saved state: %s\n", savename);
        }
    }


    
    // save ecg to folder
    sprintf(savefile, "ecg.csv");
    strcpy(savename, foldname);
    strcat(savename, savefile);
    printf("writing file: %s\n", savename); 
    writeVecCsvFloat(h_ecg, savename, 0, ecg_save_size);

    if (spiral_track_flag==1)
    {
        sprintf(savefile, "tiptrack_%d.csv", count_tip_save++);
        strcpy(savename, foldname_track);
        strcat(savename, savefile);
        printf("writing file: %s\n", savename); 
        cudaMemcpy(h_tip_vec, d_tip_vec, tip_vecsize*sizeof(int), cudaMemcpyDeviceToHost);
        writeVecCsvLongLongInt(h_tip_vec, savename, 0, tip_vecsize);   
    }

    clock_t finish = clock();
    printf("It took %f seconds\n", (float)(finish - start) / CLOCKS_PER_SEC);


    return 0;
}