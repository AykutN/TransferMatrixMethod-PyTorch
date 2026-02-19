pwd


    %d1p = values(1);
    %d2p = values(2);
    %d3p = values(3);
    
    % AVT ve CRI_ext hesaplamasını yap
    %[AVT, CRI_ext] = calculationTMMforPhyton(d1p, d2p, d3p);
    
    % Python'a AVT ve CRI_ext değerlerini gönder
    %fwrite(server, sprintf('%.6f,%.6f\n', AVT, CRI_ext));
