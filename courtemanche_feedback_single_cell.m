clear

%% constants
Na_o = 140;
K_o = 5.4;
Ca_o = 1.8;

R = 8.3143;
T = 310;
F = 96.4867;
Cm = 100; % pF
K_Q10 = 3;
Km_Na_i = 10;
Km_K_o = 1.5;
K_mNa = 87.5;
K_mCa = 1.38;
K_sat = 0.1;
gamma = 0.35;
K_rel = 30;
tau_tr = 180;
I_up_max = 0.005;
K_up = 0.00092;
Ca_up_max = 15;
CMDN_max = 0.05;
TRPN_max = 0.07;
CSQN_max = 10;
Km_CMDN = 0.00238;
Km_TRPN = 0.0005;
Km_CSQN = 0.8;
V_cell = 20100;
V_i =  V_cell.*0.680000;
tau_f_Ca = 2.00000;
sigma =  (1.00000./7.00000).*(exp(Na_o./67.3000) - 1.00000);
tau_u = 8.00000;
V_rel =  0.00480000.*V_cell;
V_up =  0.0552000.*V_cell;

g_Na = 7.8;
g_K1 = 0.09;
g_to = 0.1652;
g_kur_scale = 0.005;
g_Kr = 0.029411765;
g_Ks = 0.12941176;
g_B_Na = 0.0006744375;
g_B_K = 0;
g_B_Ca = 0.001131;
i_NaK_max = 0.59933874;
i_CaP_max = 0.275;
I_NaCa_max = 1600;
g_Ca_L = 0.12375;

%% time and space
time = 3600e3*5; %s *~8 = run time; 1hr sim = ~10hr run
dt = 1e-1; %s 1e-4 def
t_n = time/dt; 


%% initial conditions
V = -81.18;
Na_i = 11.17;
K_i = 139;
Cai = 1.013e-4;
m = 2.908e-3;
h = 9.649e-1;
j = 9.775e-1;
oa = 3.043e-2;
oi = 9.992e-1;
ua = 4.966e-3;
ui = 9.986e-1;
xr = 3.296e-5;
xs = 1.869e-2;
d = 1.367e-4;
f = 9.996e-1;
f_Ca = 7.755e-1;
Ca_rel = 1.488;
u = 2.35e-112;
v = 1;
w = 0.9992;
Ca_up = 1.488;


%% conductances k's
start_scale = 1;

k_na = 1*start_scale;

k_k1 = 1*start_scale;
k_ca = 1*start_scale;

k_ryr = 1*start_scale;
k_serca = 1*start_scale;
k_naca = 1*start_scale;

m_k_D = 1*start_scale;
m_k_na = 1*start_scale;

tau_g = 20e3;
tau_g_x_scale = 75*tau_g;


tau_k_na = 1*tau_g_x_scale;
%ca_tgt
Ca_tgt = 0.43; %0.43 - steady@1000


%% save data
k_save = 1;
data_step = 10;
save_size = fix(t_n./data_step);

data_Cai_syst = zeros(1,save_size);
data_Cai_dia = zeros(1,save_size);
% data_Ca_i = zeros(N_cells,N_cells,save_size);



stim_time_1 = 1e3; %s
stim_time_2 = .75e3;  %vector .1:0.2:1.5
stim_time_3 = .5e3; %s
stim_time_4 = .25e3; %s
stim_time_5 =  1e3; %s
stim_change_int = 3600e3;

k_Ca_ver = 1;


peak_flag = 0;
peak_int_flag = 0;
peak_timer = 1;
peak_time_thresh = 1750e2; %ms


%buffer remodeling
buffer_var_flag = 0;

Ca_tgt_flag = 0;
%syst/dia catgt
if Ca_tgt_flag == 1
    Ca_tgt_1 = 1.0498;
    Ca_tgt_2 = 0.3914;
end


tic
%% main for
for t = 1:t_n
    if rem(t,1000000) == 0
        disp(string(t) + "/"+string(t_n)+" " + string(t./t_n))
    end
    

    k_k1   =  -10.0 * (k_na) + 11.0;
    k_ito  =  5.0 * (k_na) - 4.0;
    k_kur  =  5.0 * (k_na) - 4.0;
    k_ks   =  -10.0 * (k_na) + 11.0;
    
    k_ca    =  5.0 * (k_na) - 4.0;
    k_ryr   =  -10.0 * (k_na) + 11.0;
    k_serca =  2.5 * (k_na) - 1.5;
    k_naca  =  -4.0 * (k_na) + 5.0;
    
    %ion potentials
    E_K =  (( R.*T)./F).*log(K_o./K_i);
    E_Na =  (( R.*T)./F).*log(Na_o./Na_i);
    E_Ca =  (( R.*T)./( 2.0.*F)).*log(Ca_o./Cai);
    
    %iNa
    alpha_m = ( 0.320000.*(V+47.1300))./(1.00000 - exp(- 0.100000.*(V+47.1300)));
    beta_m =  0.0800000.*exp( - V./11.0000);
    m_inf = alpha_m./(alpha_m+beta_m);
    tau_m = 1.00000./(alpha_m+beta_m);
    
    a=1-1./(1+exp(-(V+40)/0.24));
    alpha_h =  a.*(0.135000.*exp((V+80.0000)./ - 6.80000));
    beta_h =   a.*(3.56000.*exp( 0.0790000.*V)+ 310000..*exp( 0.350000.*V)) + (1-a).*(1.00000./( 0.130000.*(1.00000+exp((V+10.6600)./ - 11.1000))));
    alpha_j =  a.*(( (  - 127140..*exp( 0.244400.*V) -  3.47400e-05.*exp(  - 0.0439100.*V)).*(V+37.7800))./(1.00000+exp( 0.311000.*(V+79.2300))));
    beta_j =   a.*(( 0.121200.*exp(  - 0.0105200.*V))./(1.00000+exp(  - 0.137800.*(V+40.1400)))) + (1-a).*((0.300000.*exp(  - 2.53500e-07.*V))./(1.00000+exp(- 0.100000.*(V+32.0000))));

    h_inf = alpha_h./(alpha_h+beta_h);
    tau_h = 1.00000./(alpha_h+beta_h);

    j_inf = alpha_j./(alpha_j+beta_j);
    tau_j = 1.00000./(alpha_j+beta_j);
    
    diff_h = (h_inf - h)./tau_h; 
    diff_j = (j_inf - j)./tau_j;
    

    i_Na =  k_na * g_Na.*power(m, 3.0).*h.*j.*(V - E_Na);
    
    %IK1
    i_K1 = k_k1 * g_K1.*(V - E_K)./(1.0+exp( 0.07.*(V+80.0)));


    %Ito
    alpha_oa =  0.650000.*power(exp((V -  - 10.0000)./ - 8.50000)+exp(((V -  - 10.0000) - 40.0000)./ - 59.0000),  - 1.00000);
    beta_oa =  0.650000.*power(2.50000+exp(((V -  - 10.0000)+72.0000)./17.0000),  - 1.00000);
    tau_oa = power(alpha_oa+beta_oa,  - 1.00000)./K_Q10;
    oa_infinity = power(1.00000+exp(((V -  - 10.0000)+10.4700)./ - 17.5400),  - 1.00000);
    diff_oa = (oa_infinity - oa)./tau_oa;

    alpha_oi = power(18.5300+ 1.00000.*exp(((V -  - 10.0000)+103.700)./10.9500),  - 1.00000);
    beta_oi = power(35.5600+ 1.00000.*exp(((V -  - 10.0000) - 8.74000)./ - 7.44000),  - 1.00000);
    tau_oi = power(alpha_oi+beta_oi,  - 1.00000)./K_Q10;
    oi_infinity = power(1.00000+exp(((V -  - 10.0000)+33.1000)./5.30000),  - 1.00000);
    diff_oi = (oi_infinity - oi)./tau_oi;
    
    i_to =  k_ito * g_to.*power(oa, 3.00000).*oi.*(V - E_K);
    
    %IKur
    alpha_ua =  0.650000.*power(exp((V -  - 10.0000)./ - 8.50000)+exp(((V -  - 10.0000) - 40.0000)./ - 59.0000),  - 1.00000);
    beta_ua =  0.650000.*power(2.50000+exp(((V -  - 10.0000)+72.0000)./17.0000),  - 1.00000);
    tau_ua = power(alpha_ua+beta_ua,  - 1.00000)./K_Q10;
    ua_infinity = power(1.00000+exp(((V -  - 10.0000)+20.3000)./ - 9.60000),  - 1.00000);
    diff_ua = (ua_infinity - ua)./tau_ua;
    
    alpha_ui = power(21.0000+ 1.00000.*exp(((V -  - 10.0000) - 195.000)./ - 28.0000),  - 1.00000);
    beta_ui = 1.00000./exp(((V -  - 10.0000) - 168.000)./ - 16.0000);
    tau_ui = power(alpha_ui+beta_ui,  - 1.00000)./K_Q10;
    ui_infinity = power(1.00000+exp(((V -  - 10.0000) - 109.450)./27.4800),  - 1.00000);
    diff_ui = (ui_infinity - ui)./tau_ui;
    
    g_Kur = g_kur_scale .* (1 + 10./(1.00000+exp((V - 15.0000)./ - 13.0000)));
    i_Kur = k_kur * g_Kur.*power(ua, 3.00000).*ui.*(V - E_K);
    
    %IKr
    alpha_xr =  ( 0.000300000.*(V+14.1000))./(1.00000 - exp((V+14.1000)./ - 5.00000));
    beta_xr =  ( 7.38980e-05.*(V - 3.33280))./(exp((V - 3.33280)./5.12370) - 1.00000);
    tau_xr = power(alpha_xr+beta_xr,  - 1.00000);
    xr_infinity = power(1.00000+exp((V+14.1000)./ - 6.50000),  - 1.00000);

    diff_xr = (xr_infinity - xr)./tau_xr;
    
    i_Kr =  g_Kr.*xr.*(V - E_K)./(1.0+exp((V+15.0)./22.4000));
    
    %IKs
    alpha_xs =  ( 4.00000e-05.*(V - 19.9000))./(1.00000 - exp((V - 19.9000)./ - 17.0000));
    beta_xs =  ( 3.50000e-05.*(V - 19.9000))./(exp((V - 19.9000)./9.00000) - 1.00000);
    tau_xs =  0.500000.*power(alpha_xs+beta_xs,  - 1.00000);
    xs_infinity = power(1.00000+exp((V - 19.9000)./ - 12.7000),  - 0.500000);
    diff_xs = (xs_infinity - xs)./tau_xs;
    
    i_Ks =  k_ks * g_Ks.*power(xs, 2.0).*(V - E_K);
    
    %ICaL
    f_Ca_infinity = power(1.00000+Cai./0.000350000,  - 1.00000);
    diff_f_Ca = (f_Ca_infinity - f_Ca)./tau_f_Ca;
    
    d_infinity = power(1.00000+exp((V+10.0000)./ - 8.00000),  - 1.00000);
    tau_d = (1.00000 - exp((V+10.0000)./ - 6.24000))./( 0.0350000.*(V+10.0000).*(1.00000+exp((V+10.0000)./ - 6.24000)));
    diff_d = (d_infinity - d)./tau_d;
    
    f_infinity = exp( - (V+28.0000)./6.90000)./(1.00000+exp( - (V+28.0000)./6.90000));
    tau_f =  9.00000.*power( 0.0197000.*exp(  - power(0.0337000, 2.00000).*power(V+10.0000, 2.00000))+0.0200000,  - 1.00000);
    diff_f = (f_infinity - f)./tau_f;
    
    i_Ca_L = k_Ca_ver * k_ca * g_Ca_L.*d.*f.*f_Ca.*(V - 65.0);
    
    %INaK
    f_NaK = power(1.00000+ 0.124500.*exp((  - 0.100000.*F.*V)./( R.*T))+ 0.0365000.*sigma.*exp((  - F.*V)./( R.*T)),  - 1.00000);
    i_NaK = ( (( i_NaK_max.*f_NaK.*1.00000)./(1.00000+power(Km_Na_i./Na_i, 1.50000))).*K_o)./(K_o+Km_K_o);

    %NCX
    i_NaCa = k_naca * ( I_NaCa_max.*( exp(( gamma.*F.*V)./( R.*T)).*power(Na_i, 3.00000).*Ca_o -  exp(( (gamma - 1.00000).*F.*V)./( R.*T)).*power(Na_o, 3.00000).*Cai))./( (power(K_mNa, 3.00000)+power(Na_o, 3.00000)).*(K_mCa+Ca_o).*(1.00000+ K_sat.*exp(( (gamma - 1.00000).*V.*F)./( R.*T))));

    %Background
    i_B_Na =  g_B_Na.*(V - E_Na);
    i_B_K =  g_B_K.*(V - E_K);
    i_B_Ca =  g_B_Ca.*(V - E_Ca);

    %%%intracel calcium
    %calcium pump
    i_CaP = ( i_CaP_max.*Cai)./(0.000500000+Cai);
    
    %JSR ca release
    i_rel =   K_rel.*power(u, 2.00000).*v.*w.*(Ca_rel - Cai);
    Fn =  1000.00.*( 1.00000e-15.*V_rel.*i_rel -  (1.00000e-15./( 2.00000.*F)).*( 0.500000.*Cm*i_Ca_L -  0.200000.*Cm*i_NaCa));
    
    u_infinity = power(1.00000+exp( - (Fn - 3.41750e-13)./1.36700e-15),  - 1.00000);
    diff_u = (u_infinity - u)./(tau_u);
    
    tau_w =  ( 6.00000.*(1.00000 - exp( - (V - 7.90000)./5.00000)))./( (1.00000+ 0.300000.*exp( - (V - 7.90000)./5.00000)).*1.00000.*(V - 7.90000));
    w_infinity = 1.00000 - power(1.00000+exp( - (V - 40.0000)./17.0000),  - 1.00000);
    diff_w = (w_infinity - w)./tau_w;
    
    tau_v = 1.91000+ 2.09000.*power(1.00000+exp( - (Fn - 3.41750e-13)./1.36700e-15),  - 1.00000);
    v_infinity = 1.00000 - power(1.00000+exp( - (Fn - 6.83500e-14)./1.36700e-15),  - 1.00000);
    diff_v = (v_infinity - v)./tau_v;

    %transfer nsr-jsr
    i_tr = (Ca_up - Ca_rel)./tau_tr;

    % NSR uptake and leak
    i_up =  k_serca * I_up_max./(1.00000+K_up./Cai);%k_serca *
    i_up_leak = k_ryr *( I_up_max.*Ca_up)./Ca_up_max;

    %Ca release/uptake
    diff_Ca_rel = (i_tr - i_rel).*power(1.00000+( ( (1-buffer_var_flag) + k_ca * (buffer_var_flag) ).* CSQN_max.*Km_CSQN)./power(Ca_rel+Km_CSQN, 2.00000),- 1.00000);
    diff_Ca_up = i_up - (i_up_leak+( i_tr.*V_rel)./V_up);
    
    %Ca_i changes
    B1 = Cm*( 2.*i_NaCa - (i_CaP+i_Ca_L+i_B_Ca))./( 2.00000.*V_i.*F)+( V_up.*(i_up_leak - i_up)+ i_rel.*V_rel)./V_i;
    B2 = 1.00000+( ( (1-buffer_var_flag) + k_ca * (buffer_var_flag) ).* TRPN_max.*Km_TRPN)./power(Cai+Km_TRPN, 2.00000)+( ( (1-buffer_var_flag) + k_ca * (buffer_var_flag) ).* CMDN_max.*Km_CMDN)./power(Cai+Km_CMDN, 2.00000);
    diff_Ca_i = B1./B2;
    
    %ion changes Na, K
    diff_K_i = ( 2.0.*i_NaK - (i_K1+i_to+i_Kur+i_Kr+i_Ks+i_B_K))./( V_i.*F/Cm);
    diff_Na_i = (  - 3.0.*i_NaK - ( 3.0.*i_NaCa+i_B_Na+i_Na))./( V_i.*F/Cm);
    

    %integrate all 0.8839
    Cai_o = Cai;
    
    m = m_inf + (m - m_inf) .* exp(-dt./tau_m);
    h = h + dt.*diff_h;
    j = j + dt.*diff_j;
    oa = oa + dt.*diff_oa;
    oi = oi + dt.*diff_oi;
    ua = ua + dt.*diff_ua;
    ui = ui + dt.*diff_ui;
    xr = xr + dt.*diff_xr;
    xs = xs + dt.*diff_xs;
    u = u + dt.*diff_u;
    f_Ca = f_Ca + dt.*diff_f_Ca;
    d = d + dt.*diff_d;
    f = f + dt.*diff_f;
    w = w + dt.*diff_w;
    v = v + dt.*diff_v;
    Ca_rel = Ca_rel + dt.*diff_Ca_rel;
    Ca_up = Ca_up + dt.*diff_Ca_up;
    Cai = Cai + dt.*diff_Ca_i;
    K_i = K_i + dt.*diff_K_i.*0;
    Na_i = Na_i + dt.*diff_Na_i.*0;
    
    Cai_n = Cai;

    %check Ca
    if ~isreal(Cai)
        disp('Complex Cai');
        break
    end
    
    
    dCai = Cai_n - Cai_o;


    
    % current and voltages; currents in pA/pF * uF = uA/uF * uF = uA
    %stim current
    
    if mod(t,stim_time_1/dt)>=0 && mod(t,stim_time_1/dt)<=(5/dt) && (t>5)   && (t*dt<stim_change_int*1) %5 in ms 
        i_st = -10;
    elseif mod(t,stim_time_2/dt)>=0 && mod(t,stim_time_2/dt)<=(5/dt) && (t*dt>stim_change_int*1) && (t*dt<stim_change_int*2)
        i_st = -10;
    elseif mod(t,stim_time_3/dt)>=0 && mod(t,stim_time_3/dt)<=(5/dt) && (t*dt>stim_change_int*2) && (t*dt<stim_change_int*3)
        i_st = -10;
    elseif mod(t,stim_time_4/dt)>=0 && mod(t,stim_time_4/dt)<=(5/dt) && (t*dt>stim_change_int*3) && (t*dt<stim_change_int*4)
        i_st = -10;
    elseif mod(t,stim_time_5/dt)>=0 && mod(t,stim_time_5/dt)<=(5/dt) && (t*dt>stim_change_int*4) %&& (t*dt<5000e3)
        i_st = -10;
    else
        i_st = 0;
    end

    i_tot =  i_Na + i_K1 + i_to + i_Kur + i_Kr + i_Ks + i_B_Na + i_B_K + i_B_Ca + i_NaK + i_CaP + i_NaCa + i_Ca_L + i_st;
    V =  V + dt.*(- i_tot);
    
    %check V
    if isnan(V(isnan(V)))
        disp('NaN V');
        break
    end
    
    % variable conductances

    Cai_norm = Cai/6e-4;
    

    if Ca_tgt_flag == 1
        if dCai > 2e-8 && peak_flag == 0
            peak_flag = 1;
            peak_int_flag = 1;
        end

        if peak_flag == 1
            peak_timer = peak_timer + 1;
        end

        if abs(dCai<2e-10) && peak_int_flag == 1
            peak_time_thresh = peak_timer*3;
            peak_int_flag = 0;
        end

        if peak_timer > peak_time_thresh
            peak_flag = 0;
            peak_timer = 0;
        end

        if peak_flag == 1
            Ca_tgt = Ca_tgt_1;
        else
            Ca_tgt = Ca_tgt_2;
        end
    end
        
    [k_na,m_k_na] = variable_cond(k_na,m_k_na,tau_k_na,dt,Ca_tgt,Cai_norm,tau_g);

    if rem(t,data_step) == 0
        if peak_flag == 1
            data_Cai_syst(k_save) = Cai;
        else
            data_Cai_dia(k_save) = Cai;
        end
        data_u(k_save) = V;
        data_cai(k_save) = Cai;    
        data_k_na(k_save) = k_na;
        data_m_k_na(k_save) = m_k_na;

        k_save = k_save+1;
    end
end

toc



%% plot
data_cai = data_cai .* 1e6;
data_k_na = data_k_na(1:10000:end);


data_k_k1   =  -10.0 * (data_k_na) + 11.0 + 0.02;
data_k_ks   =  -10.0 * (data_k_na) + 11.0 + 0.01;
data_k_ryr   = -10.0 * (data_k_na) + 11.0 - 0.01;

data_k_ito  =  5.0 * (data_k_na) - 4.0 + 0.015;
data_k_kur  =  5.0 * (data_k_na) - 4.0 - 0.015;
data_k_ca    =  5.0 * (data_k_na) - 4.0;

data_k_serca =  2.5 * (data_k_na) - 1.5;
data_k_naca  =  -4.0 * (data_k_na) + 5.0;
    
time_x_k = (1:length(data_k_na)).*1e1./3600;
time_x_cai = (1:length(data_cai)).*1e-3./3600;

fig_save = figure;
fig_save.Position = [0 0 1000 1200];
t = tiledlayout(1,2);
t.TileSpacing = 'compact';
t.Padding = 'compact';
fsize = 16;
legend_fsize = 12;

nexttile([1,1])
% newcolors_log = ['#003049';'#f25c54';'#f79d65';'#55a630';'#f79d65';'#f7b267';'#55a630';'#2b9348';'#007f5f'];  
% colororder(newcolors_log)

plot(time_x_k,data_k_na,'Linewidth',2)
hold on
plot(time_x_k,data_k_ca,'Linewidth',2,'color',[85, 166, 48]./256)
plot(time_x_k,data_k_ito,':','Linewidth',2)
plot(time_x_k,data_k_kur,'--','Linewidth',2)
plot(time_x_k,data_k_k1,':','Linewidth',2)
plot(time_x_k,data_k_ks,'Linewidth',2)
plot(time_x_k,data_k_ryr,'Linewidth',2)
plot(time_x_k,data_k_serca,'Linewidth',2,'color',[239, 71, 111]./256)
plot(time_x_k,data_k_naca,'Linewidth',2)

xline(1,':', 'LineWidth',2)
xline(2,':', 'LineWidth',2)
xline(3,':', 'LineWidth',2)
xline(4,':', 'LineWidth',2)
box off
xlim([0,5]);
ylim([0.33,2.35])
set(gca,'FontSize',fsize,'TickDir','out')
legend('\gamma_{Na}','\gamma_{CaL}','\gamma_{ito}','\gamma_{Kur}','\gamma_{K1}','\gamma_{Ks}',...
       '\gamma_{RyR_{leak}}','\gamma_{SERCA}','\gamma_{NCX}','location','best')
legend boxoff
xlabel('time (h)')
ylabel('\gamma_{x}')
yticks(0.5:0.5:2)


nexttile([1,1])
movmean_cai = movmean(data_cai,100000);
plot(time_x_cai,data_cai,'color',[1, 0, 0 ,.75], 'LineWidth',2)
hold on
plot(time_x_cai,movmean_cai,'color',[.1, .1, .1, .75], 'LineWidth',2)
yline(0.43*600,'--', 'LineWidth',2)
xline(1,':', 'LineWidth',2)
xline(2,':', 'LineWidth',2)
xline(3,':', 'LineWidth',2)
xline(4,':', 'LineWidth',2)
box off
xlim([0,5]);
ylim([0,800]);
set(gca,'FontSize',fsize,'TickDir','out')
xlabel('time (h)')
ylabel('Ca_i (nM)')




%%

function [g_x,m_g_x] = variable_cond(g_x,m_g_x,tau_g_x,dt,Ca_tgt,Cai,tau_g)
    m_g_x = m_g_x + (Ca_tgt - Cai).*dt./tau_g_x;
    m_g_x(m_g_x<0) = 0;
    g_x = g_x + (m_g_x - g_x).*dt./tau_g;
    g_x(g_x<0) = 0;
end


