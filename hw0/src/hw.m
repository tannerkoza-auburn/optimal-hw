%% Homework 0 - Optimal Estimation - Tanner Koza

clear
clc
close all

%% Problem #1

% Part A
% Simple Rotation Table EOM: J*theta_double_dot + b*theta_dot = Tau

J = 10; % Rotational Moment of Inertia (kg*m^2)
b = 1; % Rotational Damping (N*m-s/rad)

% Part B
A = [0 1; 0 -b/J]; % Dynamic Matrix
B = [0; 1/J]; % Control Matrix
C = [1 0]; % Measurement Matrix
D = 0;

sys_ss = ss(A,B,C,D); % Open Loop System SS
sys_tf = tf(sys_ss); % Open Loop System TF

% Part C
s_sys = eig(sys_ss); % Open Loop System Eigenvalues

%% Problem #2

% Part A
ob = obsv(A,C); % Observability Matrix
observability = rank(ob);

% Part B
wn_ob = 100 * pi; % Desired Natural Frequency (rad/s)
zeta_ob = 0.7; % Desired Damping Ratio
s_des_ob = [-wn_ob*zeta_ob - (wn_ob*sqrt(1-zeta_ob^2)*1i), ...
           -wn_ob*zeta_ob + (wn_ob*sqrt(1-zeta_ob^2)*1i)]; % Desired Observer Eigenvalues

L = place(A',C',s_des_ob)'; % Observer Gain Matrix

% Part C
A_ob = A - L*C; % Observer Closed Loop A Matrix
ob_ss = ss(A_ob,L,C,D); % Observer SS

dt = 1/1000; % Simulation Time Step & Time Vector (Sampling (s))
t = 0:dt:0.05;

u = zeros(1,length(t));
y = zeros(1, length(t));
x = zeros(2,length(t));
dx = zeros(2,length(t));
xhat = zeros(2,length(t));
dxhat = zeros(2,length(t));

x(:,1) = [10; 0]; % Initialized State & State Estimate
xhat(:,1) = [0; 0];

for i = 1:length(t) - 1

    y(i) = C*x(:,i); % Actual State "Measurements"

    dx(:,i) = A*x(:,i); % Actual State Derivative

    x(:,i+1) = x(:,i) + dx(:,i)*dt; % Integrate Actual States



    dxhat(:,i) = A*xhat(:,i) + B*u(i); % Time Update

    dxhat(:,i) = dxhat(:,i) + L*(y(i) - C*xhat(:,i)); % Measurement Update

    xhat(:,i+1) = xhat(:,i) + dxhat(:,i)*dt; % Integrate Estimated States

end

% Plotting
figure

subplot(2,1,1)
plot(t, x(1,:)', t, xhat(1,:)') % Plot Actual & Estimated States
hold on
plot(t, x(1,:)' - xhat(1,:)') % Plot Error

title("Observer: Angular Position & Estimate")
xlabel('time (s)')
ylabel('position (rads)')
legend('\theta','\theta_{est}','error','Location','southeast')

subplot(2,1,2)
plot(t, dx(1,:)', t, dxhat(1,:)')
hold on
plot(t, dx(1,:)' - dxhat(1,:)')

title("Observer: Angular Velocity Estimate")
xlabel('time (s)')
ylabel('velocity (rads/s)')
legend('\omega','\omega_{est}','error','Location','southeast')

%% Problem #3

% Part A
co = ctrb(A,B); % Observability Matrix
controllability = rank(co); 

% Part B
wn_co = 20 * pi; % Desired Natural Frequency (rad/s)
zeta_co = 0.7; % Desired Damping Ratio
s_des_co = [-wn_co*zeta_co - (wn_co*sqrt(1-zeta_co^2)*1i), ...
           -wn_co*zeta_co + (wn_co*sqrt(1-zeta_co^2)*1i)]; % Desired Controller Eigenvalues

K = place(A,B,s_des_co); % Controller Gain Matrix

% Part C
A_co = A - B*K;
co_ss = ss(A_co,B,C,D); % Controller SS

dt = 0.00001; % Simulation Time Step & Time Vector
t = 0:dt:0.25;

u = zeros(1,length(t));
y = zeros(1, length(t));
x = zeros(2,length(t));
dx = zeros(2,length(t));
xhat = zeros(2,length(t));
dxhat = zeros(2,length(t));

x(:,1) = [0; 0]; % Initialized State & State Estimate
xhat(:,1) = [1; 0];
r = [1; 0]; % Reference 

for i = 1:length(t) - 1

    u(i) = K*(r -xhat(:,i)); % Control Input

    y(i) = C*x(:,i); % Actual State "Measurements"

    dx(:,i) = A*x(:,i) + B*u(i); % Actual State Derivative

    x(:,i+1) = x(:,i) + dx(:,i)*dt; % Integrate Actual States



    dxhat(:,i) = A*xhat(:,i) + B*u(i); % Time Update

    dxhat(:,i) = dxhat(:,i) + L*(y(i) - C*xhat(:,i)); % Measurement Update

    xhat(:,i+1) = xhat(:,i) + dxhat(:,i)*dt; % Integrate Estimated States

end

% Plotting
figure

subplot(2,1,1)
plot(t, x(1,:)', t, xhat(1,:)') % Plot Actual & Estimated States
hold on
plot(t, x(1,:)' - xhat(1,:)') % Plot Error

title("Observer + Controller: Angular Position & Estimate")
xlabel('time (s)')
ylabel('position (rads)')
legend('\theta','\theta_{est}','error','Location','southeast')

subplot(2,1,2)
plot(t, dx(1,:)', t, dxhat(1,:)')
hold on
plot(t, dx(1,:)' - dxhat(1,:)')

title("Observer + Controller: Angular Velocity Estimate")
xlabel('time (s)')
ylabel('velocity (rads/s)')
legend('\omega','\omega_{est}','error','Location','southeast')

%% Problem #5

% Part A
[A_z, B_z, C_z, D_z] = c2dm(A, B, C, D, dt); % Continuous to Discrete

sysz_ss = ss(A_z,B_z,C_z,D_z); % Discretized Open Loop SS
sysz_tf = tf(sysz_ss); % Discretized Open Loop TF

z_sys = eig(A_z); % Discretized Open Loop Eigenvalues

% Part B
z_des_ob = exp(s_des_ob*dt); % Discretized Observer Eigenvalues

L_z = place(A_z', C_z', z_des_ob)'; % Discretized Observer Gain Matrix

% Part C
z_des_co = exp(s_des_co*dt); % Discretized Controller Eigenvalues

K_z = place(A_z, B_z, z_des_co); % Discretized Controller Gain Matrix

% Part E
A_compz = A_z - B_z*K_z - L_z*C_z;
B_compz = L_z;
C_compz = K_z;
D_compz = 0;

compz_ss = ss(A_compz,B_compz,C_compz,D_compz);
% [compz_num, compz_den] = ss2tf(A_compz,B_compz,C_compz,D_compz);
% compz_tf = tf(compz_num, compz_den)

compz_cl = feedback(compz_ss*sysz_ss, 1);

z_compz_cl = eig(compz_cl);

%% Problem #6

% Plotting
figure

plot(t, x(1,:)', t, xhat(1,:)') % Plot Actual & Estimated States
hold on

u = zeros(1,length(t));
y = zeros(1, length(t));
x = zeros(2,length(t));
dx = zeros(2,length(t));
xhat = zeros(2,length(t));
dxhat = zeros(2,length(t));

x(:,1) = [0; 0]; % Initialized State & State Estimate
xhat(:,1) = [1; 0];

for i = 1:length(t) - 1

    u(i) = K_z*(r -xhat(:,i)); % Control Input

    y(i) = C_z*x(:,i); % Actual State "Measurements"

    x(:,i+1) = A_z*x(:,i) + B_z*u(i); % Actual State Derivative
    


    xhat(:,i+1) = A_z*xhat(:,i) + B_z*u(i) + L_z*(y(i) - C_z*xhat(:,i)); % Measurement Update

end

plot(t, x(1,:)', t, xhat(1,:)') % Plot Actual & Estimated States

title("Compensator: Angular Position & Estimate")
xlabel('time (s)')
ylabel('position (rads)')
legend('\theta','\theta_{est}','Location','southeast')

