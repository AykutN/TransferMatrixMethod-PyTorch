function [Absorptance,E,Reflectance] = Layer_Selective_Absorptance(wavelength,thicknesses,N,angle)

h = 4.13566766225e-15; %Planck's constant, h. [eVs]
c = 2.9979e8; % Speed of light in vacuum. [m/s]
eps_0 = 1; %Dielectric constant. [unitless]
mu_0 = 4*pi()*10^(-7); %Permeability of vacuum. [N/A^2]
beta_0_E = (eps_0/mu_0)^(1/2);
beta_0_M = (mu_0/eps_0)^(1/2);

eps_l = N.^2; %Dielectric of material. [unitless]

beta_l_E = (eps_0.*eps_l./mu_0).^(1/2);
beta_l_M = (mu_0./(eps_0.*eps_l)).^(1/2);

theta = angle*pi()/180;
n = length(thicknesses); %Number of layers
d = thicknesses; %Thicknesses of layers [m]
wL = wavelength; %Wavelength space [m]
E = h*c./wL; %Energy space [eV]
omega = 2*pi().*E./h; %Angular frequency space [s^-1]

for w = 1:length(wL)
    
    %Angle of incidence
    theta_l(1) = theta;
    if n <= 2
        theta_l(2) = asin(N(w,1)./N(w,2).*sin(theta_l(1)));
    else
        for j=2:(n-1)
            theta_l(j) = asin(N(w,j-1)./N(w,j).*sin(theta_l(j-1)));
        end
        theta_l(n) = asin(N(w,n-1)./N(w,n).*sin(theta_l(n-1)));
    end
    
    %Wave vector
    k0 = omega(w)/c;
    kx = k0.*sin(theta);
    kz = ((k0^2).*eps_l(w,:)-kx.^2).^(1/2);
    kappaz_E = kz; %TE
    kappaz_M = kz./eps_l(w,:); %TM
    
    %Propagation matrix
    for l = 1:n
        P(:,:,l) = [exp(-1i.*kz(l).*d(l)) 0; 0 exp(1i.*kz(l).*d(l))];
    end
    
    %Dynamic matrix
    for l = 1:n-1
        D_E(:,:,l) = 0.5*[1+(kappaz_E(l+1)./kappaz_E(l)) 1-(kappaz_E(l+1)./kappaz_E(l)); 1-(kappaz_E(l+1)./kappaz_E(l)) 1+(kappaz_E(l+1)./kappaz_E(l))];
        D_M(:,:,l) = 0.5*[1+(kappaz_M(l+1)./kappaz_M(l)) 1-(kappaz_M(l+1)./kappaz_M(l)); 1-(kappaz_M(l+1)./kappaz_M(l)) 1+(kappaz_M(l+1)./kappaz_M(l))];
    end
    
    %Transfer matrix between the lth layer and the substrate
    T_l_E = zeros(2,2,n-1);
    T_l_E(:,:,n-1) = P(:,:,n-1)*D_E(:,:,n-1);
    T_l_M = zeros(2,2,n-1);
    T_l_M(:,:,n-1) = P(:,:,n-1)*D_M(:,:,n-1);
    for j1 = n-2:-1:1
        T_l_E(:,:,j1) = P(:,:,j1)*D_E(:,:,j1)*T_l_E(:,:,j1+1);
        T_l_M(:,:,j1) = P(:,:,j1)*D_M(:,:,j1)*T_l_M(:,:,j1+1);
    end
    
    %Transfer matrix of the whole medium
    T_tot_E = D_E(:,:,1);
    T_tot_M = D_E(:,:,1);
    for j = 2:n-1
        T_tot_E = T_tot_E*P(:,:,j)*D_E(:,:,j);
        T_tot_M = T_tot_M*P(:,:,j)*D_M(:,:,j);
    end
    
    %Constants
    A_0 = 1;
    A_sub_E = 1/T_tot_E(1,1);
    A_sub_M = 1/T_tot_M(1,1);
    B_0_E = T_tot_E(2,1)/T_tot_E(1,1);
    B_0_M = T_tot_M(2,1)/T_tot_M(1,1);
    for l = 2:n-1
        %
        A_l_E = T_l_E(1,1,l)/T_tot_E(1,1);
        B_l_E = T_l_E(2,1,l)/T_tot_E(1,1);
        A_l_M = T_l_M(1,1,l)/T_tot_M(1,1);
        B_l_M = T_l_M(2,1,l)/T_tot_M(1,1);
        
        %Poynting vectors
        S_0_f_E = beta_0_E*cos(theta);
        S_0_b_E = beta_0_E*(abs(B_0_E))^2*cos(theta);
        S_0_f_M = beta_0_M*cos(theta);
        S_0_b_M = beta_0_M*(abs(B_0_M))^2*cos(theta);
        S_l_f_E = beta_l_E(w,l).*(abs(A_l_E))^2*cos(theta_l(l));
        S_l_b_E = beta_l_E(w,l).*(abs(B_l_E))^2*cos(theta_l(l));
        S_l_f_M = beta_l_M(w,l).*(abs(A_l_M))^2*cos(theta_l(l));
        S_l_b_M = beta_l_M(w,l).*(abs(B_l_M))^2*cos(theta_l(l));
        
        %Absorptance
        Absorptance_E(w,l) = ((S_0_f_E-S_0_b_E)-(S_l_f_E-S_l_b_E))/S_0_f_E;
        Absorptance_M(w,l) = ((S_0_f_M-S_0_b_M)-(S_l_f_M-S_l_b_M))/S_0_f_M;
        Absorptance(w,l) = (Absorptance_E(w,l)+Absorptance_M(w,l))/2;
    end
    
    %Reflectance
    Reflectance_E(w) = S_0_b_E/S_0_f_E;
    Reflectance_M(w) = S_0_b_M/S_0_f_M;
    Reflectance(w) = (Reflectance_E(w)+Reflectance_M(w))/2;
    Reflectance = Reflectance';
    Absorptance=Absorptance';
end 
end