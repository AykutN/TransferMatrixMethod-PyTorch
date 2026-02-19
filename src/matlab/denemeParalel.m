% MATLAB TCP/IP Server
server = tcpip('0.0.0.0', 12345, 'NetworkRole', 'server', 'Terminator', 'LF', 'Timeout', 30);
fopen(server);
disp('MATLAB server is ready.');

while true
    % Python'dan veri al
    data = fscanf(server);
    
    % Gelen veriyi kontrol et
    if isempty(data)
        disp('No data received. Skipping iteration.');
        continue; % Veri yoksa döngünün bu adımını atla
    end
    
    disp(['Received: ', data]);
    
    % d1p, d2p, d3p değerlerini ayır
    values = str2double(split(data, ','));
    
    % Gelen veriyi kontrol et
    if any(isnan(values)) || length(values) < 3
        disp('Invalid data received. Skipping iteration.');
        continue; % Geçersiz veri varsa döngünün bu adımını atla
    end
    
    d1p = values(1);
    d2p = values(2);
    d3p = values(3);
    
    % AVT ve CRI_ext hesaplamasını yap
    [AVT, CRI_ext] = calculationTMMforPhyton(d1p, d2p, d3p);
    
    % Python'a AVT ve CRI_ext değerlerini gönder
    fwrite(server, sprintf('%.6f,%.6f\n', AVT, CRI_ext));
end

fclose(server);
disp('MATLAB Server shut down.');
